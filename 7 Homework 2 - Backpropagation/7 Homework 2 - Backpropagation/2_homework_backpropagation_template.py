import random
import math
import numpy as np


class NeuralNetwork:
    def __init__(self, hidden_layer_weights, output_layer_weights, activation_type='poly'):
        self.lr = 0.5  # learning rate → how big the weight update step is

        # Create hidden layer → takes input from x1, x2
        self.hidden_layer = NeuronLayer(hidden_layer_weights, activation_type)

        # Create output layer → takes input from hidden layer outputs
        self.output_layer = NeuronLayer(output_layer_weights, activation_type)

    def feed_forward(self, input):
        # Step 1: pass original inputs through hidden layer
        # input = [1, 1] → hidden layer → [4, 9]
        hidden_layer_output = self.hidden_layer.feed_forward(input)

        # Step 2: hidden layer OUTPUT becomes input to output layer
        # [4, 9] → output layer → [289, 16]
        output_layer_output = self.output_layer.feed_forward(hidden_layer_output)

        return output_layer_output

    def compute_delta(self, target):

        # ============================================================
        # PART 1: Output Layer Delta
        # Formula: δ_o = (output - target) × f'(net)
        # ============================================================

        # Create empty array to store output deltas
        # e.g: [0.0, 0.0] for 2 output neurons
        self.dE_dO_net = np.zeros(len(self.output_layer.neurons))

        # Loop through each output neuron
        for o, o_node in enumerate(self.output_layer.neurons):

            # Step 1: how wrong is this neuron?
            # e.g: o=0 → 289 - 290 = -1
            dE_dO_out = o_node.output - target[o]

            # Step 2: derivative of activation at this neuron
            # poly: f'(net) = 2 × net
            # e.g: o=0 → 2 × 17 = 34
            d_out_d_net = o_node.activation_derv()

            # Step 3: multiply both → final delta for this output neuron
            # e.g: o=0 → -1 × 34 = -34
            self.dE_dO_net[o] = dE_dO_out * d_out_d_net
            print(f'Delta o[{o}]: {self.dE_dO_net[o]}')

        # ============================================================
        # PART 2: Hidden Layer Delta
        # Formula: δ_h = (Σ δ_o × w_ho) × f'(net_h)
        # ============================================================

        # Create empty array to store hidden deltas
        # e.g: [0.0, 0.0] for 2 hidden neurons
        self.dE_dH_net = np.zeros(len(self.hidden_layer.neurons))

        # Loop through each hidden neuron
        for h, h_node in enumerate(self.hidden_layer.neurons):

            # Collect responsibility from ALL output neurons
            dE_dH_out = 0

            for o, to_node in enumerate(self.output_layer.neurons):
                # weight connecting this hidden neuron h → output neuron o
                # e.g: h=0, o=0 → weight = 2 (from output_weights row 0, col 0)
                d_net_d_out = to_node.weights[h]

                # add: δ_o × w_ho
                # e.g: h=0 → (-34 × 2) + (16 × 1) = -52
                dE_dH_out += self.dE_dO_net[o] * d_net_d_out

            # derivative of activation at this hidden neuron
            # poly: f'(net) = 2 × net
            # e.g: h=0 → 2 × 2 = 4
            d_out_d_net = h_node.activation_derv()

            # final delta for this hidden neuron
            # e.g: h=0 → -52 × 4 = -208
            self.dE_dH_net[h] = dE_dH_out * d_out_d_net

            # ✅ FIX: print INSIDE loop → prints each hidden delta separately
            # ❌ WRONG was: print outside loop → prints all at once as array
            print(f'Delta h[{h}]: {self.dE_dH_net[h]}')

    def update_weights(self):

        # ============================================================
        # PART 1: Update Output Layer Weights
        # Formula: new_w = old_w - lr × δ_o × input_to_that_weight
        # ============================================================

        for o, o_node in enumerate(self.output_layer.neurons):
            for h, weight in enumerate(o_node.weights):

                # dE/dW = delta × input that came into this weight
                # input to output layer = hidden layer outputs [4, 9]
                # e.g: o=0, h=0 → -34 × 4 = -136
                dE_dW = self.dE_dO_net[o] * o_node.input[h]

                # update: new_w = old_w - 0.5 × dE_dW
                # e.g: 2 - (0.5 × -136) = 2 + 68 = 70
                weight -= self.lr * dE_dW
                print(f'node o: {o} - w_ho: {h}: Delata {dE_dW} => new w = {weight}')

        # ============================================================
        # PART 2: Update Hidden Layer Weights
        # Formula: new_w = old_w - lr × δ_h × input_to_that_weight
        # ============================================================

        for h, h_node in enumerate(self.hidden_layer.neurons):
            for i, weight in enumerate(h_node.weights):

                # dE/dW = delta × original input [x1, x2]
                # e.g: h=0, i=0 → -208 × 1 = -208
                dE_dW = self.dE_dH_net[h] * h_node.input[i]

                # update: new_w = old_w - 0.5 × dE_dW
                # e.g: 1 - (0.5 × -208) = 1 + 104 = 105
                weight -= self.lr * dE_dW
                print(f'node h: {h} - w_ih: {i}: Delata {dE_dW} => new w = {weight}')

    def train_step(self, input, target):
        # Step 1: Forward pass → get predictions
        output = self.feed_forward(input)
        print('network output:', output)

        # Step 2: Backward pass → calculate deltas
        self.compute_delta(target)

        # Step 3: Update all weights
        self.update_weights()


class NeuronLayer:
    def __init__(self, weights, activation_type):
        # weights.shape[0] = number of rows = number of neurons
        # e.g: weights = [[1,1],[2,1]] → shape[0] = 2 → 2 neurons

        # Step 1: create empty slots
        # e.g: [None, None] for 2 neurons
        self.neurons = [None] * weights.shape[0]

        # Step 2: fill each slot with a Neuron
        # enumerate gives (index, row) for each row in weights
        for layer_node, prev_node_weights in enumerate(weights):
            # e.g: layer_node=0, prev_node_weights=[1,1]
            #      → create Neuron([1,1]) and put in slot 0
            # e.g: layer_node=1, prev_node_weights=[2,1]
            #      → create Neuron([2,1]) and put in slot 1
            self.neurons[layer_node] = Neuron(prev_node_weights, activation_type)

    def feed_forward(self, inputs):
        # Empty list to collect each neuron's output
        outputs = []

        for neuron in self.neurons:
            # ✅ FIX: use 'inputs' (the parameter passed in)
            # ❌ WRONG was: neuron.calc_net_out(input)
            #    'input' is Python's built-in function → returns None!
            # ✅ CORRECT: neuron.calc_net_out(inputs)
            #    'inputs' is the actual list [1,1] or [4,9] passed in
            res = neuron.calc_net_out(inputs)

            # collect the output
            # e.g: after neuron 0 → outputs = [4]
            # e.g: after neuron 1 → outputs = [4, 9]
            outputs.append(res)

        # return all outputs → becomes input to next layer
        return outputs


class Neuron:
    def __init__(self, weights, activation_type):
        self.weights = weights            # e.g: [1, 1] for neuron 0
        self.activation_type = activation_type  # 'poly' or 'sigmoid'

    def calc_net_out(self, input):
        # Save input → needed later in update_weights
        # e.g: input = [1, 1] for hidden layer
        # e.g: input = [4, 9] for output layer
        self.input = input

        # net = dot product of weights and inputs
        # e.g: neuron 0 → (1×1) + (1×1) = 2
        # e.g: neuron 1 → (2×1) + (1×1) = 3
        self.net = np.dot(self.weights, self.input)

        # apply activation function
        # e.g: poly → 2² = 4
        self.output = self.activation(self.net)

        return self.output

    def activation(self, net):
        # Polynomial: f(net) = net²
        if self.activation_type == 'poly':
            return net ** 2

        # Sigmoid: f(net) = 1 / (1 + e^(-net))
        if self.activation_type == 'sigmoid':
            return 1 / (1 + math.exp(-net))

        # Identity: f(net) = net (no change)
        return net

    def activation_derv(self):
        # Polynomial derivative: f'(net) = 2 × net
        if self.activation_type == 'poly':
            return 2 * self.net

        # Sigmoid derivative: f'(net) = output × (1 - output)
        if self.activation_type == 'sigmoid':
            return self.output * (1 - self.output)

        # Identity derivative: f'(net) = 1
        return 1






def poly():     # 2 x 2 x 2
    hidden_layer_weights = np.array([[1, 1],
                                     [2, 1]])
    output_layer_weights = np.array([[2, 1],
                                     [1, 0]])

    nn = NeuralNetwork(hidden_layer_weights, output_layer_weights, 'poly')

    nn.train_step([1, 1], [290, 14])

    '''
    network output: [289, 16]
    Delta o[0]: -34.0
    Delta o[1]: 16.0
    Delta h[0]: -208.0
    Delta h[1]: -204.0
    node o: 0 - w_ho: 0: Delata -136.0 => new w = 70.0
    node o: 0 - w_ho: 1: Delata -306.0 => new w = 154.0
    node o: 1 - w_ho: 0: Delata 64.0 => new w = -31.0
    node o: 1 - w_ho: 1: Delata 144.0 => new w = -72.0
    node h: 0 - w_ih: 0: Delata -208.0 => new w = 105.0
    node h: 0 - w_ih: 1: Delata -208.0 => new w = 105.0
    node h: 1 - w_ih: 0: Delata -204.0 => new w = 104.0
    node h: 1 - w_ih: 1: Delata -204.0 => new w = 103.0
    '''


def sigm():     # 2 4 3
    hidden_layer_weights = np.array([[0.1, 0.1],      # 4x2 NOT 2x4
                                     [0.2, 0.1],
                                     [0.1, 0.3],
                                     [0.5, 0.01]])

    output_layer_weights = np.array([[0.1, 0.2, 0.1, 0.2],
                                     [0.1, 0.1, 0.1, 0.5],
                                     [0.1, 0.4, 0.3, 0.2]])

    nn = NeuralNetwork(hidden_layer_weights, output_layer_weights, 'sigmoid')

    nn.train_step([1, 2], [0.4, 0.7, 0.6])

    '''
    network output: [0.5913212667539777, 0.6219200057374265, 0.6508562785102494]
    Delta o[0]: 0.04623477887224621
    Delta o[1]: -0.01835937944358026
    Delta o[2]: 0.011556701931083076
    Delta h[0]: 0.000963950492482261
    Delta h[1]: 0.0028912254002713203
    Delta h[2]: 0.001386714367431997
    Delta h[3]: 0.000556197739142091
    node o: 0 - w_ho: 0: Delata 0.026559222739603632 => new w = 0.0867203886301982
    node o: 0 - w_ho: 1: Delata 0.027680191578841717 => new w = 0.18615990421057915
    node o: 0 - w_ho: 2: Delata 0.030893513891333994 => new w = 0.08455324305433301
    node o: 0 - w_ho: 3: Delata 0.028996038295713737 => new w = 0.18550198085214314
    node o: 1 - w_ho: 0: Delata -0.010546408134670482 => new w = 0.10527320406733524
    node o: 1 - w_ho: 1: Delata -0.010991533920193718 => new w = 0.10549576696009687
    node o: 1 - w_ho: 2: Delata -0.01226751284879592 => new w = 0.10613375642439797
    node o: 1 - w_ho: 3: Delata -0.01151404380893776 => new w = 0.5057570219044689
    node o: 2 - w_ho: 0: Delata 0.006638660943333523 => new w = 0.09668066952833325
    node o: 2 - w_ho: 1: Delata 0.006918854837737182 => new w = 0.39654057258113146
    node o: 2 - w_ho: 2: Delata 0.007722046916941944 => new w = 0.29613897654152904
    node o: 2 - w_ho: 3: Delata 0.007247759802026145 => new w = 0.19637612009898694
    node h: 0 - w_ih: 0: Delata 0.000963950492482261 => new w = 0.09951802475375887
    node h: 0 - w_ih: 1: Delata 0.001927900984964522 => new w = 0.09903604950751775
    node h: 1 - w_ih: 0: Delata 0.0028912254002713203 => new w = 0.19855438729986435
    node h: 1 - w_ih: 1: Delata 0.005782450800542641 => new w = 0.09710877459972869
    node h: 2 - w_ih: 0: Delata 0.001386714367431997 => new w = 0.09930664281628401
    node h: 2 - w_ih: 1: Delata 0.002773428734863994 => new w = 0.298613285632568
    node h: 3 - w_ih: 0: Delata 0.000556197739142091 => new w = 0.49972190113042897
    node h: 3 - w_ih: 1: Delata 0.001112395478284182 => new w = 0.00944380226085791
    '''



if __name__ == '__main__':
    #poly()
    sigm()

