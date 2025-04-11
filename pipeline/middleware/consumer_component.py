from Tianyu_pipeline.pipeline.utils import sql_interface 

class consumer_component(object):
    """
    This is the super class of the consumer components.
    Consumer components have bi-directional references with the process consumer.
    Different consumer components can be implemented by inheriting this class.
    Consumer components of the same consumer can call the methods and get the attributes of other components.
    """
    def __init__(self):
        self.connected_to_consumer = False
        self.consumer = None
        self.consumer_id = None
    def connect(self,consumer):
        """
        Connect to the process consumer.
        """
        self.connected_to_consumer = True
        self.consumer = consumer
        self.consumer_id = id(consumer)
    def disconnect(self):
        """
        Disconnect from the process consumer.
        """
        self.connected_to_consumer = False
        self.consumer = None
        self.consumer_id = None
    @property
    def sql_interface(self):
        """
        Return the sql_interface of the consumer if exists.
        Setup the sql_interface of the consumer if not exists.
        """
        if self.connected_to_consumer:
            
            if hasattr(self, "_sql_interface"):
                #release this attribute
                del self._sql_interface
            return self.consumer.sql_interface
        else:
            self._sql_interface = sql_interface.sql_interface()
            return self._sql_interface
