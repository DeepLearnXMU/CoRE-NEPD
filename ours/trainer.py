import argparse
import json
import logging
import os
import random
import shutil
import sys
import pdb
# import apex
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from data_helper import BlkPosInterface, SimpleListDataset
import pickle


class ContextError(Exception):
    def __init__(self):
        pass


class Once:
    def __init__(self, rank):
        self.rank = rank

    def __enter__(self):
        if self.rank > 0:
            sys.settrace(lambda *args, **keys: None)
            frame = sys._getframe(1)
            frame.f_trace = self.trace
        return True

    def trace(self, frame, event, arg):
        raise ContextError

    def __exit__(self, type, value, traceback):
        if type == ContextError:
            return True
        else:
            return False


class OnceBarrier:
    def __init__(self, rank):
        self.rank = rank

    def __enter__(self):
        if self.rank > 0:
            sys.settrace(lambda *args, **keys: None)
            frame = sys._getframe(1)
            frame.f_trace = self.trace
        return True

    def trace(self, frame, event, arg):
        raise ContextError

    def __exit__(self, type, value, traceback):
        if self.rank >= 0:
            torch.distributed.barrier()
        if type == ContextError:
            return True
        else:
            return False


class Cache:
    def __init__(self, rank):
        self.rank = rank

    def __enter__(self):
        if self.rank not in [-1, 0]:
            torch.distributed.barrier()
        return True

    def __exit__(self, type, value, traceback):
        if self.rank == 0:
            torch.distributed.barrier()
        return False


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


class Prefetcher:
    def __init__(self, dataloader, stream):
        self.dataloader = dataloader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        self.iter = iter(self.dataloader)
        self.preload()
        return self

    def preload(self):
        try:
            self.next = next(self.iter)
        except StopIteration:
            self.next = None
            return
        with torch.cuda.stream(self.stream):
            next_list = list()
            for v in self.next:
                if type(v) == torch.Tensor:
                    next_list.append(v.cuda(non_blocking=True))
                else:
                    next_list.append(v)
            self.next = tuple(next_list)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next is not None:
            result = self.next
            self.preload()
            return result
        else:
            raise StopIteration

    def __len__(self):
        return len(self.dataloader)


class TrainerCallback:
    def __init__(self):
        pass

    def on_argument(self, parser):
        pass

    def load_model(self):
        pass

    def load_data(self):
        pass

    def collate_fn(self):
        return None, None, None

    def on_train_epoch_start(self, epoch):
        pass

    def on_train_step(self, step, train_step, inputs, extra, loss, outputs):
        pass

    def on_train_epoch_end(self, epoch):
        pass

    def on_dev_epoch_start(self, epoch):
        pass

    def on_dev_step(self, step, inputs, extra, outputs):
        pass

    def on_dev_epoch_end(self, epoch):
        pass

    def on_test_epoch_start(self, epoch):
        pass

    def on_test_step(self, step, inputs, extra, outputs):
        pass

    def on_test_epoch_end(self, epoch):
        pass

    def process_train_data(self, data):
        pass

    def process_dev_data(self, data):
        pass

    def process_test_data(self, data):
        pass

    def on_save(self, path):
        pass

    def on_load(self, path):
        pass


class Trainer:
    def __init__(self, callback: TrainerCallback):
        self.callback = callback
        self.callback.trainer = self
        logging.basicConfig(level=logging.INFO)

    def parse_args(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--train', action='store_true')
        self.parser.add_argument('--dev', action='store_true')
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--debug', action='store_true')
        self.parser.add_argument("--per_gpu_train_batch_size", default=1, type=int)
        self.parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int)
        self.parser.add_argument("--learning_rate", default=5e-5, type=float)
        self.parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
        self.parser.add_argument("--weight_decay", default=0.0, type=float)
        self.parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        self.parser.add_argument("--max_grad_norm", default=1.0, type=float)
        self.parser.add_argument("--epochs", default=10, type=int)
        self.parser.add_argument("--warmup_ratio", default=0.1, type=float)
        self.parser.add_argument("--logging_steps", type=int, default=500)
        self.parser.add_argument("--save_steps", type=int, default=10000)
        self.parser.add_argument("--seed", type=int, default=42)
        self.parser.add_argument("--num_workers", type=int, default=0)
        self.parser.add_argument("--local_rank", type=int, default=-1)
        self.parser.add_argument("--fp16", action="store_true")
        self.parser.add_argument("--fp16_opt_level", type=str, default="O1")
        self.parser.add_argument("--no_cuda", action="store_true")
        self.parser.add_argument("--load_checkpoint", default=None, type=str)
        self.parser.add_argument("--load_classi_checkpoint", default=None, type=str)
        self.parser.add_argument("--ignore_progress", action='store_true')
        self.parser.add_argument("--dataset_ratio", type=float, default=1.0)
        self.parser.add_argument("--no_save", action="store_true")
        self.parser.add_argument("--intro_save", default="../data/", type=str)
        # self.parser.add_argument("--model_name", default="bert", type=str)
        self.callback.on_argument(self.parser)
        self.args = self.parser.parse_args()
        keys = list(self.args.__dict__.keys())
        for key in keys:
            value = getattr(self.args, key)
            if type(value) == str and os.path.exists(value):
                setattr(self.args, key, os.path.abspath(value))
        if not self.args.train:
            self.args.epochs = 1
        self.train = self.args.train
        self.dev = self.args.dev
        self.test = self.args.test
        self.debug = self.args.debug
        self.per_gpu_train_batch_size = self.args.per_gpu_train_batch_size
        self.per_gpu_eval_batch_size = self.args.per_gpu_eval_batch_size
        self.learning_rate = self.args.learning_rate
        self.gradient_accumulation_steps = self.args.gradient_accumulation_steps
        self.weight_decay = self.args.weight_decay
        self.adam_epsilon = self.args.adam_epsilon
        self.max_grad_norm = self.args.max_grad_norm
        self.epochs = self.args.epochs
        self.warmup_ratio = self.args.warmup_ratio
        self.logging_steps = self.args.logging_steps
        self.save_steps = self.args.save_steps
        self.seed = self.args.seed
        self.num_workers = self.args.num_workers
        self.local_rank = self.args.local_rank
        self.fp16 = self.args.fp16
        self.fp16_opt_level = self.args.fp16_opt_level
        self.no_cuda = self.args.no_cuda
        self.load_checkpoint = self.args.load_checkpoint
        # self.load_classi_checkpoint = self.args.load_checkpoint
        self.ignore_progress = self.args.ignore_progress
        self.dataset_ratio = self.args.dataset_ratio
        self.no_save = self.args.no_save
        self.callback.args = self.args
        # self.model_name = self.args.model_name
        self.intro_save = self.args.intro_save

    def set_env(self):
        if self.debug:
            sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)
        if self.local_rank == -1 or self.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            self.n_gpu = 0 if self.no_cuda else torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.local_rank)
            device = torch.device("cuda", self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.n_gpu = 1
        set_seed(self.seed, self.n_gpu)
        self.device = device
        with self.once_barrier():
            if not os.path.exists('r'):
                os.mkdir('r')
            runs = os.listdir('r')
            i = max([int(c) for c in runs], default=-1) + 1
            os.mkdir(os.path.join('r', str(i)))
            src_names = [source for source in os.listdir() if source.endswith('.py')]
            for src_name in src_names:
                shutil.copy(src_name, os.path.join('r', str(i), src_name))
            os.mkdir(os.path.join('r', str(i), 'output'))
            os.mkdir(os.path.join('r', str(i), 'tmp'))
        runs = os.listdir('r')
        i = max([int(c) for c in runs])
        os.chdir(os.path.join('r', str(i)))
        with self.once_barrier():
            json.dump(sys.argv, open('output/args.json', 'w'))
        logging.info("Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            self.local_rank, device, self.n_gpu, bool(self.local_rank != -1), self.fp16))
        # self.train_batch_size = self.per_gpu_train_batch_size
        # self.eval_batch_size = self.per_gpu_eval_batch_size
        self.train_batch_size = self.per_gpu_train_batch_size * max(1, self.n_gpu)
        self.eval_batch_size = self.per_gpu_eval_batch_size * max(1, self.n_gpu)

        # if self.fp16:
        #     apex.amp.register_half_function(torch, "einsum")
        self.stream = torch.cuda.Stream()

    def set_model(self):
        self.model = self.callback.load_model()
        k = 0
        # pdb.set_trace()
        # for name, parameters in self.model.named_parameters():
        #     #   print(name)
        #     # if "predictor_balanced" in name:
        #     #     parameters.requires_grad=True
        #     # else:
        #     #     parameters.requires_grad=False
        #     if parameters.requires_grad==True:
        #         if "sen_bert" in name:
        #             continue
        #         l = 1
        #         for j in parameters.size():
        #             l *= j
        #         #l += parameters.size().item()
        #         k = k+l
        #         print(name, parameters.size())
        # print("total", k )
        # pdb.set_trace()
        #     for
        # params = list(self.model.parameters())
        # k = 0
        # for i in params:
        #     l = 1
        #     #print("该层的结构：" + str(list(i.size())))
        #     if i.requires_grad == True:
        #         print("该层的结构：", i, str(list(i.size())))
        #         for j in i.size():
        #             l *= j
        #     #print("该层参数和：" + str(l))
        #     k = k + l
        # print("总参数数量和：" + str(k))
        # sss
        self.model.to(self.device)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        # if self.fp16:
        #     self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level=self.fp16_opt_level)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        if self.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank],
                                                                   output_device=self.local_rank,
                                                                   find_unused_parameters=True)

    def once(self):
        return Once(self.local_rank)

    def once_barrier(self):
        return OnceBarrier(self.local_rank)

    def cache(self):
        return Cache(self.local_rank)

    def load_data(self):
        self.train_step = 1
        self.epochs_trained = 0
        self.steps_trained_in_current_epoch = 0
        self.intro_train_step = 1
        ##this may cause an error, you should use the absolute path
        relations = json.load(open('../data/rawdata/relations.json'))
        relations.sort()
        self.relations = ['n/a'] + relations
        self.relation2id = dict()
        for index, relation in enumerate(self.relations):
            self.relation2id[relation] = index

        train_dataset, dev_dataset, test_dataset = self.callback.load_data()
        # train_dataset, dev_dataset = self.callback.load_data()
        train_fn, dev_fn, test_fn = self.callback.collate_fn()
        if train_dataset:
            if self.dataset_ratio < 1:
                train_dataset = torch.utils.data.Subset(train_dataset,
                                                        list(range(int(len(train_dataset) * self.dataset_ratio))))
            self.train_dataset = train_dataset

            target = []
            for d in self.train_dataset:
                if d[-1] == 'o':
                    target.append(self.relation2id[d[1]])
                else:
                    tmpd = dict()
            
                    for dd in d:
                        if dd[-1] not in tmpd:
                            tmpd[dd[-1]] = 0
                        tmpd[dd[-1]]+=1
                    inverse = [(value, key) for key, value in tmpd.items()]
                    id = max(inverse)[0]
                    if id in self.relation2id:
                        target.append(self.relation2id[id])
                    else:
                        target.append(0)
            # # target = train_dataset.targets
            # class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
            # weight = 1. / class_sample_count
            # samples_weight = np.array([weight[t] for t in target])
            # samples_weight = torch.from_numpy(samples_weight)
            # # samples_weight = samples_weight.double()
            # sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

            self.train_sampler = RandomSampler(self.train_dataset) if self.local_rank == -1 else DistributedSampler(
                self.train_dataset)
            # self.train_sampler = sampler if self.local_rank == -1 else DistributedSampler(
            #     self.train_dataset)
            self.train_dataloader = Prefetcher(
                DataLoader(self.train_dataset, sampler=self.train_sampler, batch_size=self.train_batch_size,
                           collate_fn=train_fn, num_workers=self.num_workers), self.stream)

            self.t_total = len(self.train_dataloader) // self.gradient_accumulation_steps * self.epochs
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=int(self.t_total * self.warmup_ratio),
                                                             num_training_steps=self.t_total)
        if dev_dataset:
            if self.dataset_ratio < 1:
                dev_dataset = torch.utils.data.Subset(dev_dataset,
                                                      list(range(int(len(dev_dataset) * self.dataset_ratio))))
            self.dev_dataset = dev_dataset
            self.dev_sampler = SequentialSampler(self.dev_dataset) if self.local_rank == -1 else DistributedSampler(
                self.dev_dataset)
            self.dev_dataloader = Prefetcher(
                DataLoader(self.dev_dataset, sampler=self.dev_sampler, batch_size=self.eval_batch_size,
                           collate_fn=dev_fn, num_workers=self.num_workers), self.stream)
        if test_dataset:
            if self.dataset_ratio < 1:
                test_dataset = torch.utils.data.Subset(test_dataset,
                                                       list(range(int(len(test_dataset) * self.dataset_ratio))))
            self.test_dataset = test_dataset
            self.test_sampler = SequentialSampler(self.test_dataset) if self.local_rank == -1 else DistributedSampler(
                self.test_dataset)
            self.test_dataloader = Prefetcher(
                DataLoader(self.test_dataset, sampler=self.test_sampler, batch_size=self.eval_batch_size,
                           collate_fn=test_fn, num_workers=self.num_workers), self.stream)

    def restore_checkpoint(self, path, ignore_progress=False):
        if self.no_save:
            return
        model_to_load = self.model.module if hasattr(self.model, "module") else self.model
        if "*" in path:
            path, path_classi = path.split("*")
            dict_classi = torch.load(os.path.join(path_classi, 'pytorch_model.bin'), map_location=self.device)
            pre_dict = torch.load(os.path.join(path, 'pytorch_model.bin'), map_location=self.device)

            for key,value in dict_classi.items():
            # print(key)
                if "predictor_balanced" in key:
                    tmpkey = key.split(".")[1]
                    pre_dict["predictor_balanced."+tmpkey] = value
        else:
            pre_dict = torch.load(os.path.join(path, 'pytorch_model.bin'), map_location=self.device)
            for key,value in self.model.predictor_balanced.state_dict().items():
                # print(key)
                pre_dict["predictor_balanced."+key] = value
        model_to_load.load_state_dict(pre_dict)
        if self.train:
            self.optimizer.load_state_dict(torch.load(os.path.join(path, "optimizer.pt"), map_location=self.device))
            self.scheduler.load_state_dict(torch.load(os.path.join(path, "scheduler.pt"), map_location=self.device))
        
        self.callback.on_load(path)
        if self.train:
            if not ignore_progress:
                self.train_step = int(path.split("-")[-1])
                print("len(self.train_dataloader",len(self.train_dataloader))
                print("self.gradient_accumulation_steps",self.gradient_accumulation_steps)
                tmp = max(1, len(self.train_dataloader) // self.gradient_accumulation_steps)
                self.epochs_trained = self.train_step // tmp
                self.steps_trained_in_current_epoch = self.train_step % (tmp)
            logging.info("  Continuing training from checkpoint, will skip to saved train_step")
            logging.info("  Continuing training from epoch %d", self.epochs_trained)
            logging.info("  Continuing training from train step %d", self.train_step)
            logging.info("  Will skip the first %d steps in the first epoch", self.steps_trained_in_current_epoch)

    def save_checkpoint(self):
        if self.no_save:
            return
        output_dir = os.path.join('output', "checkpoint-{}".format(self.train_step))
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        self.callback.on_save(output_dir)
    
    def _register_embedding_list_hook(self,model, embeddings_list):
        def forward_hook(module, inputs, output):
            embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())
        embedding_layer = model.embeddings.word_embeddings
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _register_embedding_gradient_hooks(self,model, embeddings_gradients):
        def hook_layers(module, grad_in, grad_out):
            embeddings_gradients.append(grad_out[0])
        embedding_layer = model.embeddings.word_embeddings
        hook = embedding_layer.register_backward_hook(hook_layers)
        return hook
    def saliency_map(self,model, tmp, label):
        torch.enable_grad()
        model.eval()
        embeddings_list = []
        handle = self._register_embedding_list_hook(model, embeddings_list)
        embeddings_gradients = []
        hook = self._register_embedding_gradient_hooks(model, embeddings_gradients)

        model.zero_grad()
        A = model(*tmp[0])

        ##pooled_output -> last layer hiddenstates[0]
        pooled_output = A[1]


        P = torch.nn.Linear(768, 3)
        out = P(pooled_output.cpu())

        loss = torch.Tensor([0])

        sLoss = torch.nn.CrossEntropyLoss()

        loss = sLoss(out, torch.Tensor(label).long().cpu())

        loss.backward()
        
        handle.remove()
        hook.remove()

        saliency_grad = embeddings_gradients[0].cpu().detach().numpy()
        saliency_grad_final = []

        index = 0
        for saliency_grad_data in saliency_grad:

            saliency_grad_tmp = np.sum(saliency_grad_data * embeddings_list[0][index], axis=1)
            # print("saliency_grad_tmp.shape")
            # print(saliency_grad_tmp.shape)
            ###ord = 1
            norm = np.linalg.norm(saliency_grad_tmp, ord=1)

            
            saliency_grads = [e / norm for e in saliency_grad_tmp] 

            saliency_grad_final.append(saliency_grads)

            index = index + 1
        # saliency_grad = 123
        return saliency_grad_final
    
    def run(self):
        self.parse_args()
        self.set_env()
        with self.once():
            self.writer = SummaryWriter()
        self.set_model()
        self.load_data()
        if self.load_checkpoint is not None:
            self.restore_checkpoint(self.load_checkpoint, self.ignore_progress)
        best_performance = 0
        best_step = -1
        for epoch in tqdm(range(self.epochs)):
            if epoch < self.epochs_trained:
                continue
            with self.once():
                logging.info('epoch %d', epoch)
            if self.train:
                tr_loss, logging_loss = 0.0, 0.0
                self.model.zero_grad()
                self.model.train()
                self.callback.on_train_epoch_start(epoch)
                prev_loss = 0.0
                if self.local_rank >= 0:
                    self.train_sampler.set_epoch(epoch)
                print("==========Training==========")
                batch_start = next(iter(self.train_dataloader))
                #total_sens = list()
                for step, batch in enumerate(tqdm(self.train_dataloader, disable=self.local_rank > 0)):
                    if step < self.steps_trained_in_current_epoch:
                        continue
                    
                    extra, selected_inputs, selected_rets = self.callback.process_train_data(batch)

                    try:
                        outputs = self.model(**selected_inputs)

                        loss = outputs[0]
                        
                        logging.info('loss: {}'.format(loss))
                        if self.n_gpu > 1:
                            loss = loss.mean()
                        if self.gradient_accumulation_steps > 1:
                            loss = loss / self.gradient_accumulation_steps
                        if self.local_rank < 0 or (step + 1) % self.gradient_accumulation_steps == 0:

                            loss.backward()

                        else:
                            with self.model.no_sync():
    
                                loss.backward()
                        tr_loss += loss.item()
                        if (step + 1) % self.gradient_accumulation_steps == 0:
                            # if self.fp16:
                            #     torch.nn.utils.clip_grad_norm_(apex.amp.master_params(self.optimizer), self.max_grad_norm)
                            # else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            self.optimizer.step()
                            self.scheduler.step()
                            self.model.zero_grad()
                            self.train_step += 1
                            with self.once():
                                if self.train_step % self.logging_steps == 0:
                                    self.writer.add_scalar("lr", self.scheduler.get_lr()[0], self.train_step)
                                    self.writer.add_scalar("loss", (tr_loss - logging_loss) / self.logging_steps,
                                                           self.train_step)
                                    logging_loss = tr_loss
                                if self.train_step % self.save_steps == 0:
                                    self.save_checkpoint()
                        # torch.cuda.empty_cache()
                        self.callback.on_train_step(step, self.train_step, selected_inputs, extra, loss.item(), outputs)

                    except RuntimeError as e:
                        print(e)
                        continue

                with self.once():
                    self.save_checkpoint()
                self.callback.on_train_epoch_end(epoch)
            if self.dev:
                print("==========Dev==========")
                with torch.no_grad():
                    self.model.eval()
                    #total_sens = list()
                    self.callback.on_dev_epoch_start(epoch)
                    for step, batch in enumerate(tqdm(self.dev_dataloader, disable=self.local_rank > 0)):
                        try:
                            extra, selected_inputs, selected_rets = self.callback.process_dev_data(batch)
                            outputs = self.model(**selected_inputs)
                            self.callback.on_dev_step(step, selected_inputs, extra, outputs)
                        except RuntimeError as e:
                            print(e)
                            continue
                    performance = self.callback.on_dev_epoch_end(epoch)
                    if performance > best_performance:
                        best_performance = performance
                        best_step = self.train_step
        if self.test:
            print("==========Test==========")
            total_docs = []
            with torch.no_grad():
                if best_step > 0 and self.train:
                    logging.info("test_best_step{}".format(best_step))
                    self.restore_checkpoint(os.path.join('output', "checkpoint-{}".format(best_step)))
                self.model.eval()
                #total_sens = list()
                self.callback.on_test_epoch_start(epoch)
                for step, batch in enumerate(tqdm(self.test_dataloader, disable=self.local_rank > 0)):
                    extra, selected_inputs, selected_rets = self.callback.process_test_data(batch)
                    try:
                        outputs = self.model(**selected_inputs)
                    except RuntimeError as e:
                        print(e)
                        continue
                    self.callback.on_test_step(step, selected_inputs, extra, outputs)
                self.callback.on_test_epoch_end(epoch)
        with self.once():
            self.writer.close()
        json.dump(True, open('output/f.json', 'w'))

    def distributed_broadcast(self, l):
        assert type(l) == list or type(l) == dict
        if self.local_rank < 0:
            return l
        else:
            torch.distributed.barrier()
            process_number = torch.distributed.get_world_size()
            # print("type(l)", type(l))
            # print("type(l[0])", type(l[0]))
            json.dump(l, open(f'tmp/{self.local_rank}.json', 'w'))
            torch.distributed.barrier()
            objs = list()
            for i in range(process_number):
                objs.append(json.load(open(f'tmp/{i}.json')))
            if type(objs[0]) == list:
                ret = list()
                for i in range(process_number):
                    ret.extend(objs[i])
            else:
                ret = dict()
                for i in range(process_number):
                    for k, v in objs.items():
                        assert k not in ret
                        ret[k] = v
            torch.distributed.barrier()
            return ret

    def distributed_merge(self, l):
        assert type(l) == list or type(l) == dict
        if self.local_rank < 0:
            return l
        else:
            torch.distributed.barrier()
            process_number = torch.distributed.get_world_size()
            # print("type(l[0])",type(l[0]))
            json.dump(l, open(f'tmp/{self.local_rank}.json', 'w'))
            torch.distributed.barrier()
            if self.local_rank == 0:
                objs = list()
                for i in range(process_number):
                    objs.append(json.load(open(f'tmp/{i}.json')))
                if type(objs[0]) == list:
                    ret = list()
                    for i in range(process_number):
                        ret.extend(objs[i])
                else:
                    ret = dict()
                    for i in range(process_number):
                        for k, v in objs.items():
                            assert k not in ret
                            ret[k] = v
            else:
                ret = None
            torch.distributed.barrier()
            return ret

    def distributed_get(self, v):
        if self.local_rank < 0:
            return v
        else:

            torch.distributed.barrier()
            if self.local_rank == 0:
                json.dump(v, open('tmp/v.json', 'w'))
            torch.distributed.barrier()
            v = json.load(open('tmp/v.json'))
            torch.distributed.barrier()
            return v
    def distributed_get_broadcast(self,v):
        if self.local_rank < 0:
            return v
        else:
            torch.distributed.barrier()
            json.dump(v, open(f'tmp/{self.local_rank}.json', 'w'))
            torch.distributed.barrier()
            process_number = torch.distributed.get_world_size()
            objs = list()
            for i in range(process_number):
                tmpv = json.load(open(f'tmp/{i}.json'))
                objs.append(tmpv)
            if None in objs:
                v = None
            else:
                v = 1
            torch.distributed.barrier()
            return v
    def distributed_get_broadcast_loss(self,v):
        if self.local_rank < 0:
            return v
        else:
            torch.distributed.barrier()
            json.dump(v, open(f'tmp/{self.local_rank}_loss.json', 'w'))
            torch.distributed.barrier()
            process_number = torch.distributed.get_world_size()
            objs = list()
            for i in range(process_number):
                with open(f'tmp/{i}_loss.json', "r") as f:
                    try:
                        tmpv = json.load(f)
                        objs.append(tmpv)
                    except:
                        objs.append(None)
            if None in objs:
                v = None
            else:
                v = 1
            torch.distributed.barrier()
            return v
    def _write_estimation(self, buf, relevance_blk, f):
        for i, blk in enumerate(buf):
            f.write(f'{blk.pos} {relevance_blk[i].item()}\n')

    def _score_blocks(self, qbuf, relevance_token):
        ends = qbuf.block_ends()
        relevance_blk = torch.ones(len(ends), device='cpu')
        for i in range(len(ends)):
            if qbuf[i].blk_type > 0:  # query
                relevance_blk[i] = (relevance_token[ends[i - 1]:ends[i]]).mean()
        return relevance_blk

    def _collect_estimations_from_dir(self, est_dir):
        ret = {}
        for shortname in os.listdir(est_dir):
            filename = os.path.join(est_dir, shortname)
            if shortname.startswith('estimations_'):
                with open(filename, 'r') as fin:
                    for line in fin:
                        l = line.split()
                        pos, estimation = int(l[0]), float(l[1])
                        ret[pos].estimation = estimation
                os.replace(filename, os.path.join(est_dir, 'backup_' + shortname))
        return ret
