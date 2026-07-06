"""
异步处理队列模块
使用 asyncio 实现任务队列，支持并发控制和超时处理
"""
import asyncio
import time
from dataclasses import dataclass
from typing import Callable, Any, Optional
from enum import Enum
import threading


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def duration(self):
        """执行耗时"""
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 3)
        return None


class AsyncTaskQueue:
    """
    异步任务队列
    - 支持最大并发数限制
    - 支持任务超时
    - 支持任务优先级
    - 支持结果缓存
    """
    
    def __init__(self, max_workers=4, default_timeout=30):
        """
        初始化队列
        Args:
            max_workers: 最大并发工作数
            default_timeout: 默认超时时间（秒）
        """
        self.max_workers = max_workers
        self.default_timeout = default_timeout
        
        # 信号量控制并发
        self.semaphore = asyncio.Semaphore(max_workers)
        
        # 任务结果缓存
        self._results = {}
        self._results_lock = threading.Lock()
        
        # 统计信息
        self._stats = {
            'total': 0,
            'completed': 0,
            'failed': 0,
            'timeout': 0
        }

    async def submit(self, task_id: str, func: Callable, *args, 
                     timeout: Optional[float] = None, **kwargs) -> TaskResult:
        """
        提交任务到队列
        Args:
            task_id: 任务唯一标识
            func: 执行函数
            timeout: 超时时间（秒）
        Returns:
            TaskResult: 任务结果
        """
        timeout = timeout or self.default_timeout
        
        # 检查是否已有结果
        if task_id in self._results:
            cached = self._results[task_id]
            if cached.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT):
                return cached

        # 创建结果占位
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING
        )
        self._update_result(task_id, result)
        
        self._stats['total'] += 1

        # 使用信号量控制并发
        async with self.semaphore:
            result.status = TaskStatus.PROCESSING
            result.start_time = time.time()
            self._update_result(task_id, result)

            try:
                # 在线程池中执行阻塞任务
                loop = asyncio.get_event_loop()
                future = loop.run_in_executor(
                    None, 
                    lambda: func(*args, **kwargs)
                )
                
                # 等待结果或超时
                task_result = await asyncio.wait_for(future, timeout=timeout)
                
                result.status = TaskStatus.COMPLETED
                result.result = task_result
                self._stats['completed'] += 1
                
            except asyncio.TimeoutError:
                result.status = TaskStatus.TIMEOUT
                result.error = f"任务执行超时（{timeout}秒）"
                self._stats['timeout'] += 1
                
            except Exception as e:
                result.status = TaskStatus.FAILED
                result.error = str(e)
                self._stats['failed'] += 1

            finally:
                result.end_time = time.time()
                self._update_result(task_id, result)

        return result

    def _update_result(self, task_id: str, result: TaskResult):
        """更新结果缓存"""
        with self._results_lock:
            self._results[task_id] = result

    def get_result(self, task_id: str) -> Optional[TaskResult]:
        """获取任务结果"""
        return self._results.get(task_id)

    def clear_cache(self, max_age: Optional[float] = None):
        """
        清理结果缓存
        Args:
            max_age: 最大保留时间（秒），None则清理所有
        """
        with self._results_lock:
            if max_age is None:
                self._results.clear()
            else:
                now = time.time()
                to_remove = [
                    tid for tid, r in self._results.items()
                    if r.end_time and (now - r.end_time) > max_age
                ]
                for tid in to_remove:
                    del self._results[tid]

    @property
    def stats(self):
        """获取统计信息"""
        active = sum(1 for r in self._results.values() 
                    if r.status == TaskStatus.PROCESSING)
        pending = sum(1 for r in self._results.values() 
                     if r.status == TaskStatus.PENDING)
        
        return {
            **self._stats,
            'active': active,
            'pending': pending,
            'cache_size': len(self._results)
        }


# 全局队列实例（单例模式）
_queue_instance = None
_queue_lock = threading.Lock()


def get_task_queue(max_workers=4, default_timeout=30) -> AsyncTaskQueue:
    """获取全局任务队列"""
    global _queue_instance
    if _queue_instance is None:
        with _queue_lock:
            if _queue_instance is None:
                _queue_instance = AsyncTaskQueue(
                    max_workers=max_workers,
                    default_timeout=default_timeout
                )
    return _queue_instance


def reset_task_queue():
    """重置全局队列"""
    global _queue_instance
    with _queue_lock:
        _queue_instance = None