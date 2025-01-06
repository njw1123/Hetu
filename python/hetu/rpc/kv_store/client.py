import grpc
from hetu.rpc import heturpc_pb2_grpc
from hetu.rpc import heturpc_pb2
from .const import *
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '__type__': 'ndarray',
                'dtype': str(obj.dtype),
                'data': obj.tolist()
            }
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

def numpy_decoder(obj):
    if '__type__' in obj and obj['__type__'] == 'ndarray':
        return np.array(obj['data'], dtype=obj['dtype'])
    return obj

class RemoteDict:
    def __init__(self, client, dict_name):
        self.client = client
        self.dict_name = dict_name

    def put(self, key, value):
        full_key = f'{self.dict_name}{DELIMITER}{key}'
        if isinstance(value, (dict, list, np.ndarray, np.number)):
            value = json.dumps(value, cls=NumpyEncoder)
        elif not isinstance(value, str):
            raise ValueError("Value must be str, dict, list, or numpy type")
        self.client.stub.PutJson(heturpc_pb2.PutJsonRequest(key=full_key, value=value))

    def get(self, key):
        full_key = f'{self.dict_name}{DELIMITER}{key}'
        response = self.client.stub.GetJson(heturpc_pb2.GetJsonRequest(key=full_key))
        value = response.value
        try:
            value = json.loads(value, object_hook=numpy_decoder)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error, unable to parse value: {value}")
            raise e
        return value

    def remove(self, key):
        full_key = f'{self.dict_name}{DELIMITER}{key}'
        response = self.client.stub.RemoveJson(heturpc_pb2.RemoveJsonRequest(key=full_key))
        return response.message

    def get_many(self, keys):
        return {key: self.get(key) for key in keys}

class KeyValueStoreClient:
    def __init__(self, address='localhost:50051'):
        self.channel = grpc.insecure_channel(address)
        self.stub = heturpc_pb2_grpc.DeviceControllerStub(self.channel)

    def register_dict(self, dict_name):
        return RemoteDict(self, dict_name)

# example
if __name__ == '__main__':
    client = KeyValueStoreClient(address='localhost:50051')

    # Register dictionaries
    data_store = client.register_dict('test_store')

    # Store nested structures containing numpy arrays
    print("Storing nested structure...")
    nested_data = {
        "user1": {
            "profile": {
                "name": "Alice",
                "scores": np.array([95, 88, 92])
            },
            "metrics": np.array([0.5, 0.8])
        },
        "user2": {
            "profile": {
                "name": "Bob",
                "scores": np.array([75, 85, 80])
            },
            "metrics": np.array([0.7, 0.6])
        }
    }
    data_store.put('nested_data', nested_data)

    # Retrieve nested structure
    print("Retrieving nested structure...")
    retrieved_data = data_store.get('nested_data')
    print(f"Retrieved data: {retrieved_data}")
    print(f"Is 'user1->metrics' a numpy array? {isinstance(retrieved_data['user1']['metrics'], np.ndarray)}")
