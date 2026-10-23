"""Execution-only options: ordered, bounded CPU preparation and encoder chunking."""
import queue,threading,time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import torch


def validate_options(options):
    options=dict(options or {})
    if set(options)-{'encoder_chunk','prefetch_studies','omit_unused_tissue'}:raise ValueError('Unknown execution option')
    if options.get('encoder_chunk',2) not in (2,4,8,16,32):raise ValueError('Unsupported encoder chunk')
    if options.get('prefetch_studies',0) not in (0,1,2):raise ValueError('At most two future complete Studies are admitted')
    return options


class OrderedPrefetch:
    """One producer may hold ONE unconsumed Study, including construction.

    Together with the consumer's current Study, this bounds live input batches
    to two. The next Study is prepared in original order, never sampled anew.
    Source construction uses local per-Study RNG; augmentation stays on the
    consuming training thread. No model output or graph is cached.
    """
    def __init__(self,iterator,max_bytes=4*2**30):
        self.iterator=iterator;self.queue=queue.Queue();self.slots=threading.Semaphore(1);self.stop=threading.Event()
        self.max_bytes=max_bytes
        self.wait_seconds=0.;self.produce_seconds=0.;self.max_batch_bytes=0
        self.thread=threading.Thread(target=self.produce,daemon=True,name='complete-mri-prefetch');self.thread.start()

    def produce(self):
        while not self.stop.is_set():
            if not self.slots.acquire(timeout=.1):continue
            if self.stop.is_set():break
            t=time.perf_counter()
            try:value=next(self.iterator)
            except StopIteration:self.queue.put(('end',None));break
            except BaseException as exc:self.queue.put(('error',exc));break
            self.produce_seconds+=time.perf_counter()-t
            size=sum(v.numel()*v.element_size() for v in value.values() if isinstance(v,torch.Tensor))
            if size>self.max_bytes:
                self.queue.put(('error',ValueError('Complete Study exceeds the declared 4-GiB prefetch tensor budget')));break
            self.max_batch_bytes=max(self.max_batch_bytes,size)
            self.queue.put(('value',value))

    def __iter__(self):return self

    def __next__(self):
        t=time.perf_counter();kind,value=self.queue.get();self.wait_seconds+=time.perf_counter()-t;self.slots.release()
        if kind=='end':self.close();raise StopIteration
        if kind=='error':self.close();raise value
        return value

    def close(self):
        self.stop.set();self.slots.release();self.thread.join()
        while not self.queue.empty():self.queue.get_nowait()


def prune_unused(batch,model_config,options):
    if not options.get('omit_unused_tissue'):return batch
    if model_config.anatomy_auxiliary or model_config.local_evidence:raise ValueError('Cannot remove consumed tissue supervision')
    return {k:v for k,v in batch.items() if k not in ('tissue_targets','tissue_quality','tissue_pixel_weight')}


class OrderedStudyPrefetch:
    """Two independently constructed Studies, returned in frozen sampler order.

    Each future is limited to 2 GiB of tensors, so the two-future preparation
    budget is 4 GiB; the consumer's current Study is accounted separately.
    The frozen cohort's conservative largest-item bound is below 1.44 GiB.
    """
    def __init__(self,dataset,schedule):
        self.dataset=dataset;self.schedule=iter(schedule);self.pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='mri-prepare')
        self.pending=deque();self.wait_seconds=0.;self.max_batch_bytes=0
        for _ in range(2):self.submit()

    def prepare(self,index):
        from .data import collate_studies
        value=collate_studies([self.dataset[index]])
        size=sum(v.numel()*v.element_size() for v in value.values() if isinstance(v,torch.Tensor))
        if size>2*2**30:raise ValueError('Study exceeds the declared 2-GiB per-future budget')
        value={k:v.pin_memory() if torch.cuda.is_available() and isinstance(v,torch.Tensor) else v for k,v in value.items()}
        return value,size

    def submit(self):
        try:index=next(self.schedule)
        except StopIteration:return
        self.pending.append(self.pool.submit(self.prepare,index))

    def __iter__(self):return self

    def __next__(self):
        if not self.pending:self.close();raise StopIteration
        t=time.perf_counter()
        try:value,size=self.pending.popleft().result()
        except BaseException:self.close();raise
        self.wait_seconds+=time.perf_counter()-t;self.max_batch_bytes=max(self.max_batch_bytes,size);self.submit();return value

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True);self.pending.clear()
