# cache in memory or disk


from Tianyu_pipeline.pipeline.middleware.consumer_component import consumer_component
import sys
import os
class cache_manager(consumer_component):
    def __init__(self,cache_upper_limit = 5*2**30):
        super().__init__()
        self._memory_cache = {}
        self.memory_size = 0
        self._cache_dir = None
    @property
    def cache_dir(self):
        if self._cache_dir is None:
            if self.consumer is not None:
                self._cache_dir = self.consumer.fs.path_root+"/cache"
            else:
                self._cache_dir = "./"
        return self._cache_dir
    
    def in_disk(self,key):
        return os.path.exists(os.path.join(self.cache_dir,key))
    
    def delete_disk(self,key):
        if self.in_disk(key):
            os.remove(os.path.join(self.cache_dir,key))
            return True
        else:
            return False
        
    def set_memory(self,key,value,size = -1,force_clear_cache = False):
        if size == -1:
            size = sys.getsizeof(value)
        if self.memory_size + size > self.cache_upper_limit:
            if not force_clear_cache:
                raise Exception("Memory cache is full")
            else:
                #clear the cache
                self.memory_size = 0
                self._memory_cache = {}

        if self.in_memory(key):
            self.delete_memory(key)

        self._memory_cache[key] = {"value":value,"size":size}
        self.memory_size += size
        return True
    
    def in_memory(self,key):
        return key in self._memory_cache
    
    def get_memory(self,key):
        if key in self._memory_cache:
            return self._memory_cache[key]['value']
        else:
            raise Exception("Key not found in memory cache")
    
    def delete_memory(self,key):
        if self.in_memory(key):
            self.memory_size -= self._memory_cache[key]['size']
            del self._memory_cache[key]
        return
    def clear_memory(self):
        self.memory_size = 0
        self._memory_cache = {}
        return True

        
        
