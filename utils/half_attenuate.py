# def get_hidden_sizes(input_size):

#     # Define the half-attenuation law of the neuron size

#     hidden_sizes = []
#     size = input_size
#     while size > 16:
#         hidden_sizes.append(size)
#         size = size // 2
#     return hidden_sizes

class HiddenSizesGenerator:

    def __init__(self, input_size, hyperparams={}):
        self.input_size = input_size
        self.hyperparams = hyperparams

    def generate(self):
        num_layers = self.hyperparams.get("num_layers", 4)
        decay_rate = self.hyperparams.get("decay_rate", 0.5)
        
        hidden_sizes = []
        size = self.input_size
        
        for i in range(num_layers):
            hidden_sizes.append(size)
            size = int(size * decay_rate)
            
        return tuple(hidden_sizes)