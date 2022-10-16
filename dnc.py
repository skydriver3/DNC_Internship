import tensorflow as tf 
from tensorflow import keras as K 
import numpy as np 
from collections import namedtuple 

DNC_State = namedtuple("DNC_State", 
                            ["lstm_state", 
                            "read_vectors", 
                            "write_weight", 
                            "read_weight", 
                            "usage", 
                            "memory", 
                            "precedence_weight", 
                            "link"])

Memory_State = namedtuple("Memory_State" , ["write_weight", 
                                            "read_weight", 
                                            "usage", 
                                            "memory", 
                                            "precedence_weight", 
                                            "link"])

TemporalLinkageState = namedtuple('TemporalLinkageState',
                                              ('link', 'precedence_weights'))

_EPSILON = 1e-6


def reduce_prod(x, axis, name=None):
    """Efficient reduce product over axis.
    Uses tf.cumprod and tf.gather_nd as a workaround to the poor performance of calculating tf.reduce_prod's gradient on CPU.
    """
    with tf.name_scope('util_reduce_prod'):
        cp = tf.math.cumprod(x, axis, reverse=True)
        size = tf.shape(cp)[0]
        idx1 = tf.range(tf.cast(size, tf.float32), dtype=tf.float32)
        idx2 = tf.zeros([size], tf.float32)
        indices = tf.stack([idx1, idx2], 1)
        return tf.gather_nd(cp, tf.cast(indices, tf.int32))   


def batch_gather(values, indices):
    """Returns batched `tf.gather` for every row in the input."""
    with tf.name_scope('batch_gather'):
        idx = tf.expand_dims(indices, -1)
        size = tf.shape(indices)[0]
        rg = tf.range(size, dtype=tf.int32)
        rg = tf.expand_dims(rg, -1)
        rg = tf.tile(rg, [1, int(indices.get_shape()[-1])])
        rg = tf.expand_dims(rg, -1)
        gidx = tf.concat([rg, idx], -1)
        return tf.gather_nd(values, gidx)

def batch_invert_permutation(permutations):
    """Returns batched `tf.invert_permutation` for every row in `permutations`."""
    with tf.name_scope('batch_invert_permutation'):
        perm = tf.cast(permutations, tf.float32)
        dim = int(perm.get_shape()[-1])
        size = tf.cast(tf.shape(perm)[0], tf.float32)
        delta = tf.cast(tf.shape(perm)[-1], tf.float32)
        rg = tf.range(0, size * delta, delta, dtype=tf.float32)
        rg = tf.expand_dims(rg, 1)
        rg = tf.tile(rg, [1, dim])
        perm = tf.add(perm, rg)
        flat = tf.reshape(perm, [-1])
        perm = tf.math.invert_permutation(tf.cast(flat, tf.int32))
        perm = tf.reshape(perm, [-1, dim])
        return tf.subtract(perm, tf.cast(rg, tf.int32))


class DNC(K.layers.Layer) : 
    def __init__(self, lstm_units, mem_size, word_size, read_head_number, write_head_number, clip_value=None, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
        self.controller_output_size = lstm_units
        self.lstm = K.layers.LSTMCell(lstm_units)
        self.read_weight = K.layers.Dense(lstm_units, use_bias=False) 
        self.memory_size = mem_size 
        self.word_size = word_size 
        self.class_dtype = dtype 
        self.read_head_number = read_head_number
        self.write_head_number = write_head_number
        self._clip_value = clip_value 

    def build(self, input_shape):
        self.memory = Memory(self.memory_size, self.word_size, self.write_head_number, self.read_head_number, self.class_dtype)  

    def _clip_if_enabled(self, x):
        if self._clip_value > 0:
            return tf.clip_by_value(x, -self._clip_value, self._clip_value)
        else:
            return x


    def call(self, inputs, states):

        state = DNC_State(*states)
        flattened_read_vectors = tf.reshape(state.read_vectors, [-1, state.read_vectors.shape[1] * state.read_vectors.shape[2]])
        X = tf.concat([inputs, flattened_read_vectors], -1)
        #X.shape.assert_has_rank(2) 
        output, controller_state = self.lstm(
                    X,
                    state.lstm_state) 
        controller_state = tf.nest.map_structure(self._clip_if_enabled, controller_state)
        output = self._clip_if_enabled(output) 
        read_vectors, memory_state = self.memory(output, Memory_State(*state[2:])) 
        # for ix, m_state in enumerate(memory_state) : 
        #     try : 
        #         m_state.shape.assert_has_rank(len(states[ix+2].shape))
        #     except : 
        #         raise Exception(f"{DNC_State._fields[ix+2]} dimension are wrong, new_state : {m_state.shape} vs old_state : {states[ix+2]}") 

        # read_vectors = tf.reshape(read_vectors, [-1, self.word_size * self.read_head_number])
        #output.shape.assert_has_rank( len(read_vectors.shape))
        out = self.read_weight(tf.concat([output, 
                                        tf.reshape(read_vectors, [-1, self.word_size * self.read_head_number])], 
                                        axis=-1)) 
        
        return out, DNC_State(lstm_state=controller_state, 
                              read_vectors=read_vectors, 
                              write_weight=memory_state.write_weight, 
                              read_weight=memory_state.read_weight, 
                              usage=memory_state.usage, 
                              memory=memory_state.memory, 
                              precedence_weight=memory_state.precedence_weight, 
                              link=memory_state.link)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None): 
        if inputs is not None:
            batch_size = inputs.shape[0]
            dtype = inputs.dtype
        if batch_size is None or dtype is None:
            raise ValueError(
                'batch_size and dtype cannot be None while constructing initial state: '
                'batch_size={}, dtype={}'.format(batch_size, dtype))

        controller_state = self.lstm.get_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)
        read_vectors = tf.zeros([batch_size, self.read_head_number, self.word_size], dtype=dtype)
        write_weights = tf.zeros([batch_size, self.write_head_number, self.memory_size], dtype=dtype)
        read_weights = tf.zeros([batch_size, self.read_head_number, self.memory_size], dtype=dtype)
        usage = tf.zeros([batch_size, self.memory_size], dtype=dtype)
        memory = tf.zeros([batch_size, self.memory_size, self.word_size], dtype=dtype)
        precedence_weights = tf.zeros([batch_size, self.write_head_number, self.memory_size], dtype=dtype)
        link = tf.zeros([batch_size,self.write_head_number, self.memory_size, self.memory_size], dtype=dtype)
        
        state = DNC_State(
                controller_state, 
                read_vectors, 
                write_weights, 
                read_weights, 
                usage, 
                memory, 
                precedence_weights, 
                link) 
        
        return state


        

    @property
    def state_size(self): 
        return  (self.lstm.state_size ,
                ( self.read_head_number, self.word_size) ,
                (self.write_head_number, self.memory_size) ,
                (self.read_head_number * self.memory_size ),
                self.memory_size ,
                (self.memory_size , self.word_size ),
                (self.write_head_number, self.memory_size) ,
                (self.write_head_number, self.memory_size, self.memory_size))

def weighted_softmax(activations, strengths, strengths_op):
  """Returns softmax over activations multiplied by positive strengths.
  Args:
    activations: A tensor of shape `[batch_size, num_heads, memory_size]`, of
      activations to be transformed. Softmax is taken over the last dimension.
    strengths: A tensor of shape `[batch_size, num_heads]` containing strengths to
      multiply by the activations prior to the softmax.
    strengths_op: An operation to transform strengths before softmax.
  Returns:
    A tensor of same shape as `activations` with weighted softmax applied.
  """
  transformed_strengths = tf.expand_dims(strengths_op(strengths), -1)
  sharp_activations = activations * transformed_strengths
  return tf.nn.softmax(sharp_activations, axis=-1)

def _vector_norms(m):
  squared_norms = tf.reduce_sum(m * m, axis=2, keepdims=True)
  return tf.sqrt(squared_norms + _EPSILON)

class CosineWeights:
  """Cosine-weighted attention.
  Calculates the cosine similarity between a query and each word in memory, then
  applies a weighted softmax to return a sharp distribution.
  """

  def __init__(self,
               num_heads,
               word_size,
               strength_op=tf.nn.softplus,
               name='cosine_weights'):
    """Initializes the CosineWeights module.
    Args:
      num_heads: number of memory heads.
      word_size: memory word size.
      strength_op: operation to apply to strengths (default is tf.nn.softplus).
      name: module name (default 'cosine_weights')
    """
    super(CosineWeights, self).__init__()
    self._num_heads = num_heads
    self._word_size = word_size
    self._strength_op = strength_op

  def __call__(self, memory, keys, strengths):
    """Connects the CosineWeights module into the graph.
    Args:
      memory: A 3-D tensor of shape `[batch_size, memory_size, word_size]`.
      keys: A 3-D tensor of shape `[batch_size, num_heads, word_size]`.
      strengths: A 2-D tensor of shape `[batch_size, num_heads]`.
    Returns:
      Weights tensor of shape `[batch_size, num_heads, memory_size]`.
    """
    # Calculates the inner product between the query vector and words in memory.
    dot = tf.matmul(keys, memory, adjoint_b=True)

    # Outer product to compute denominator (euclidean norm of query and memory).
    memory_norms = _vector_norms(memory)
    key_norms = _vector_norms(keys)
    norm = tf.matmul(key_norms, memory_norms, adjoint_b=True)

    # Calculates cosine similarity between the query vector and words in memory.
    similarity = dot / (norm + _EPSILON)

    return weighted_softmax(similarity, strengths, self._strength_op)



class TemporalLinkage(tf.keras.layers.Layer) : 

    """Keeps track of write order for forward and backward addressing.
    This is a pseudo-RNNCore module, whose state is a pair `(link,
    precedence_weights)`, where `link` is a (collection of) graphs for (possibly
    multiple) write heads (represented by a tensor with values in the range
    [0, 1]), and `precedence_weights` records the "previous write locations" used
    to build the link graphs.
    The function `directional_read_weights` computes addresses following the
    forward and backward directions in the link graphs.
    """    
    def __init__(self, memory_size, num_writes, name='temporal_linkage'):
        """
        Construct a TemporalLinkage module.
            Args:
                memory_size: The number of memory slots.
                num_writes: The number of write heads.
                name: Name of the module.
        """
        super(TemporalLinkage, self).__init__(name=name)
        self.mem_size = memory_size
        self._num_writes = num_writes

    def call(self, write_weights, prev_state):
        
        link, precedence_weights = self._update_link_matrix(write_weights, prev_state.precedence_weights, prev_state.link)

        return TemporalLinkageState(link=link, precedence_weights=precedence_weights)

    def directional_read_weights(self, link, prev_read_weights, forward):
        """Calculates the forward or the backward read weights.
        For each read head (at a given address), there are `num_writes` link graphs
        to follow. Thus this function computes a read address for each of the
        `num_reads * num_writes` pairs of read and write heads.
        Args:
        link: tensor of shape `[batch_size, num_writes, memory_size,
            memory_size]` representing the link graphs L_t.
        prev_read_weights: tensor of shape `[batch_size, num_reads,
            memory_size]` containing the previous read weights w_{t-1}^r.
        forward: Boolean indicating whether to follow the "future" direction in
            the link graph (True) or the "past" direction (False).
        Returns:
        tensor of shape `[batch_size, num_reads, num_writes, memory_size]`
        """
        with tf.name_scope('directional_read_weights'):
            # We calculate the forward and backward directions for each pair of
            # read and write heads; hence we need to tile the read weights and do a
            # sort of "outer product" to get this.
            expanded_read_weights = tf.stack([prev_read_weights] * self._num_writes,
                                            1)
            result = tf.matmul(expanded_read_weights, link, adjoint_b=forward)
            # Swap dimensions 1, 2 so order is [batch, reads, writes, memory]:
            return tf.transpose(result, perm=[0, 2, 1, 3])
    
    def _link(self, prev_link, prev_precedence_weights, write_weights):
        """Calculates the new link graphs.
        For each write head, the link is a directed graph (represented by a matrix
        with entries in range [0, 1]) whose vertices are the memory locations, and
        an edge indicates temporal ordering of writes.
        
        Args:
            prev_link: A tensor of shape `[batch_size, num_writes, memory_size,
                memory_size]` representing the previous link graphs for each write
                head.
            prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
                memory_size]` which is the previous "aggregated" write weights for
                each write head.
            write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
                containing the new locations in memory written to.
        
        Returns:
            A tensor of shape `[batch_size, num_writes, memory_size, memory_size]`
            containing the new link graphs for each write head.
        """
        with tf.name_scope('link'):
            batch_size = tf.shape(prev_link)[0]
            write_weights_i = tf.expand_dims(write_weights, 3)
            write_weights_j = tf.expand_dims(write_weights, 2)
            prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights, 2)
            prev_link_scale = 1 - write_weights_i - write_weights_j
            new_link = write_weights_i * prev_precedence_weights_j
            link = prev_link_scale * prev_link + new_link
            # Return the link with the diagonal set to zero, to remove self-looping
            # edges.
            return tf.linalg.set_diag(
                link,
                tf.zeros(
                    [batch_size, self._num_writes, self.mem_size],
                    dtype=link.dtype))

    def _precedence_weights(self, prev_precedence_weights, write_weights):
        """Calculates the new precedence weights given the current write weights.
        The precedence weights are the "aggregated write weights" for each write
        head, where write weights with sum close to zero will leave the precedence
        weights unchanged, but with sum close to one will replace the precedence
        weights.
        Args:
            prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
                memory_size]` containing the previous precedence weights.
            write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
                containing the new write weights.
        Returns:
            A tensor of shape `[batch_size, num_writes, memory_size]` containing the
            new precedence weights.
        """
        with tf.name_scope('precedence_weights'):
            write_sum = tf.reduce_sum(write_weights, 2, keepdims=True)
            return (1 - write_sum) * prev_precedence_weights + write_weights 

    def _update_link_matrix(self, write_weights, prev_precedence_weights, prev_link) : 
        """Calculate the updated linkage state given the write weights.
        Args:
        write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
            containing the memory addresses of the different write heads.
        prev_state: `TemporalLinkageState` tuple containg a tensor `link` of
            shape `[batch_size, num_writes, memory_size, memory_size]`, and a
            tensor `precedence_weights` of shape `[batch_size, num_writes,
            memory_size]` containing the aggregated history of recent writes.
        Returns:
        A `TemporalLinkageState` tuple `next_state`, which contains the updated
        link and precedence weights.
        """
        link = self._link(prev_link, prev_precedence_weights,
                        write_weights)
        precedence_weights = self._precedence_weights(prev_precedence_weights,
                                                    write_weights)

        return link, precedence_weights


class Freeness (tf.keras.layers.Layer) : 
    """Memory usage that is increased by writing and decreased by reading.
    This module is a pseudo-RNNCore whose state is a tensor with values in
    the range [0, 1] indicating the usage of each of `memory_size` memory slots.
    The usage is:
    *   Increased by writing, where usage is increased towards 1 at the write
        addresses.
    *   Decreased by reading, where usage is decreased after reading from a
        location when free_gate is close to 1.
    The function `write_allocation_weights` can be invoked to get free locations
    to write to for a number of write heads.
    """

    def __init__(self, memory_size, name='freeness'):
        """Creates a Freeness module.
        Args:
        memory_size: Number of memory slots.
        name: Name of the module.
        """
        super(Freeness, self).__init__(name=name)
        self._memory_size = memory_size    

    def call(self, inputs, prev_write_weight, prev_read_weight, prev_usage, free_gate) : 
        """Calculates the new memory usage u_t.
        Memory that was written to in the previous time step will have its usage
        increased; memory that was read from and the controller says can be "freed"
        will have its usage decreased.
        Args:
        write_weights: tensor of shape `[batch_size, num_writes,
            memory_size]` giving write weights at previous time step.
        free_gate: tensor of shape `[batch_size, num_reads]` which indicates
            which read heads read memory that can now be freed.
        read_weights: tensor of shape `[batch_size, num_reads,
            memory_size]` giving read weights at previous time step.
        prev_usage: tensor of shape `[batch_size, memory_size]` giving
            usage u_{t - 1} at the previous time step, with entries in range
            [0, 1].
        Returns:
        tensor of shape `[batch_size, memory_size]` representing updated memory
        usage.
        """
        # Calculation of usage is not differentiable with respect to write weights.
        prev_write_weight = tf.stop_gradient(prev_write_weight)
        usage = self._usage_after_write(prev_usage, prev_write_weight)
        usage = self._usage_after_read(usage, free_gate, prev_read_weight)
        return usage
    
    def write_allocation_weights(self, usage, write_gates, num_writes):
        """Calculates freeness-based locations for writing to.
        This finds unused memory by ranking the memory locations by usage, for each
        write head. (For more than one write head, we use a "simulated new usage"
        which takes into account the fact that the previous write head will increase
        the usage in that area of the memory.)
       
        Args:

            usage: A tensor of shape `[batch_size, memory_size]` representing
                current memory usage.
            write_gates: A tensor of shape `[batch_size, num_writes]` with values in
                the range [0, 1] indicating how much each write head does writing
                based on the address returned here (and hence how much usage
                increases).
            num_writes: The number of write heads to calculate write weights for.
        
        Returns:

            tensor of shape `[batch_size, num_writes, memory_size]` containing the
                freeness-based write locations. Note that this isn't scaled by
                `write_gate`; this scaling must be applied externally.
        """
        with tf.name_scope('write_allocation_weights'):
            # expand gatings over memory locations
            write_gates = tf.expand_dims(write_gates, -1)

            allocation_weights = []
            for i in range(num_writes):
                allocation_weights.append(self._allocation(usage))
                # update usage to take into account writing to this new allocation
                usage += ((1 - usage) * write_gates[:, i, :] * allocation_weights[i])

            # Pack the allocation weights for the write heads into one tensor.
            return tf.stack(allocation_weights, axis=1)

    def _usage_after_write(self, prev_usage, write_weights):
        """Calcualtes the new usage after writing to memory.
        Args:
        prev_usage: tensor of shape `[batch_size, memory_size]`.
        write_weights: tensor of shape `[batch_size, num_writes, memory_size]`.
        Returns:
        New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        with tf.name_scope('usage_after_write'):
            # Calculate the aggregated effect of all write heads
            write_weights = 1 - reduce_prod(1 - write_weights, 1)
            return prev_usage + (1 - prev_usage) * write_weights

    def _usage_after_read(self, prev_usage, free_gate, read_weights):
        """Calcualtes the new usage after reading and freeing from memory.
        Args:
        prev_usage: tensor of shape `[batch_size, memory_size]`.
        free_gate: tensor of shape `[batch_size, num_reads]` with entries in the
            range [0, 1] indicating the amount that locations read from can be
            freed.
        read_weights: tensor of shape `[batch_size, num_reads, memory_size]`.
        Returns:
        New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        with tf.name_scope('usage_after_read'):
            free_gate = tf.expand_dims(free_gate, -1)
            free_read_weights = free_gate * read_weights
            phi = reduce_prod(1 - free_read_weights, 1, name='phi')
            return prev_usage * phi

    def _allocation(self, usage):
        r"""Computes allocation by sorting `usage`.
        This corresponds to the value a = a_t[\phi_t[j]] in the paper.
        Args:

            usage: tensor of shape `[batch_size, memory_size]` indicating current
                memory usage. This is equal to u_t in the paper when we only have one
                write head, but for multiple write heads, one should update the usage
                while iterating through the write heads to take into account the
                allocation returned by this function.
        Returns:

            Tensor of shape `[batch_size, memory_size]` corresponding to allocation.
        """
        with tf.name_scope('allocation'):
            # Ensure values are not too small prior to cumprod.
            usage = _EPSILON + (1 - _EPSILON) * usage

            nonusage = 1 - usage
            sorted_nonusage, indices = tf.nn.top_k(
                nonusage, k=self.mem_size, name='sort')
            sorted_usage = 1 - sorted_nonusage
            prod_sorted_usage = tf.math.cumprod(sorted_usage, axis=1, exclusive=True)
            sorted_allocation = sorted_nonusage * prod_sorted_usage
            inverse_indices = batch_invert_permutation(indices)

            # This final line "unsorts" sorted_allocation, so that the indexing
            # corresponds to the original indexing of `usage`.
            return batch_gather(sorted_allocation, inverse_indices)

class Memory (tf.keras.layers.Layer): 

    def __init__(self, mem_size, word_size, write_head_num, read_head_num, dtype) : 
        super().__init__()
        #assert dtype != None, "dtype can't be None in the args"
        self.mem_size = mem_size
        self._word_size = word_size
        self._num_writes = write_head_num
        self._num_reads = read_head_num  
        self.class_dtype = dtype 
        self._write_content_weight_mod = CosineWeights(self._num_writes, self._word_size)
        self._read_content_weight_mod = CosineWeights(self._num_reads, self._word_size)
        self.write_vector_linear = K.layers.Dense(self._num_writes * self._word_size, name='write_vectors')
        self.erase_vector_linear = K.layers.Dense(self._num_writes * self._word_size, name='erase_vectors')
        self.free_gate_linear = K.layers.Dense(self._num_reads, name='free_gate')
        self.allocation_gate_linear = K.layers.Dense(self._num_writes, name='allocation_gate')
        self.write_gate_linear = K.layers.Dense(self._num_writes, name='write_gate')
        self.num_read_modes = 1 + 2 * self._num_writes
        self.read_mode_linear = K.layers.Dense(self._num_reads * self.num_read_modes, name='read_mode')
        self.write_key_linear = K.layers.Dense(self._num_writes * self._word_size, name='read_mode')
        self.write_strengths_linear = K.layers.Dense(self._num_writes, name='write_strengths')
        self.read_keys_linear = K.layers.Dense(self._num_reads * self._word_size, name='read_mode')
        self.read_strength_linear = K.layers.Dense(self._num_reads, name='read_strengths')
        self._linkage = TemporalLinkage(mem_size, write_head_num) 
        self._freeness = Freeness(mem_size) 

    def update_memory(self, previous_mem_state, write_weight, e, v ) :
        E = tf.ones([self.mem_size, self._word_size], dtype=self.class_dtype)
        erase_vec = tf.matmul(write_weight, e, transpose_a=True)
        erase_tensor = E - erase_vec
        write_tensor = tf.matmul(write_weight, v, transpose_a=True)
        erased_mem = tf.math.multiply(previous_mem_state, erase_tensor)
        return  erased_mem + write_tensor

    def call  ( self, 
                inputs,   
                prev_state): 
        """Connects the MemoryAccess module into the graph.
                Args: 
                        inputs: tensor of shape `[batch_size, input_size]`. This is used to
                            control this access module.
                        prev_state: Instance of `AccessState` containing the previous state.
                Returns:
                        A tuple `(output, next_state)`, where `output` is a tensor of shape
                        `[batch_size, num_reads, word_size]`, and `next_state` is the new
                        `AccessState` named tuple at the current time t.
        """
        inputs = self._read_inputs(inputs)

        # Update usage using inputs['free_gate'] and previous read & write weights.
        usage = self._usage_vector(
            prev_write_weight=prev_state.write_weight,
            free_gate=inputs['free_gate'],
            prev_read_weight=prev_state.read_weight,
            prev_usage=prev_state.usage)
        # usage = self._freeness(
        #     inputs=None,
        #     prev_write_weight=prev_state.write_weights,
        #     free_gate=inputs['free_gate'],
        #     prev_read_weight=prev_state.read_weights,
        #     prev_usage=prev_state.usage)

        # Write to memory.
        write_weights = self._write_weights(inputs, prev_state.memory, usage)

        memory = self.update_memory(prev_state.memory, write_weights, e=inputs["erase_vectors"], v=inputs["write_vectors"])

        link, precedence_weights = self._update_link_matrix(write_weights, prev_state.precedence_weight, prev_state.link)
        # link, precedence_weights = self._linkage(write_weights, TemporalLinkageState(prev_state.link, prev_state.precedence_weight))
        # Read from memory.
        read_weights = self._read_weights(
            inputs,
            memory=memory,
            prev_read_weights=prev_state.read_weight,
            link=link)
        read_words = tf.matmul(read_weights, memory)

        return read_words, Memory_State(write_weights, read_weights, usage, memory, precedence_weights, link)
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None): 
        if inputs is not None:
            batch_size = inputs.shape[0]
            dtype = inputs.dtype
        if batch_size is None or dtype is None:
            raise ValueError(
                'batch_size and dtype cannot be None while constructing initial state: '
                'batch_size={}, dtype={}'.format(batch_size, dtype))

        write_weights = tf.zeros([batch_size, self._num_writes, self.mem_size], dtype=dtype)
        read_weights = tf.zeros([batch_size, self._num_reads, self.mem_size], dtype=dtype)
        usage = tf.zeros([batch_size, self.mem_size], dtype=dtype)
        memory = tf.zeros([batch_size, self.mem_size, self._word_size], dtype=dtype)
        precedence_weights = tf.zeros([batch_size, self._num_writes, self.mem_size], dtype=dtype)
        link = tf.zeros([batch_size,self._num_writes, self.mem_size, self.mem_size], dtype=dtype)
        
        state = Memory_State(write_weights, 
                            read_weights, 
                            usage, 
                            memory, 
                            precedence_weights, 
                            link) 
        
        return state

    def _read_inputs(self, inputs):
        """Applies transformations to `inputs` to get control for this module. `input_size` should be the length of the feature dimension of `inputs`"""

        #assert tuple(inputs.shape) == (None, input_size) , f"Input shape should be {(None, input_size)} but is {inputs.shape} vs "
        
        def _linear(first_dim, second_dim, linear_layer, name, activation=None):
            """Returns a linear transformation of `inputs`, followed by a reshape."""
            linear = linear_layer(inputs)
            if activation is not None:
                linear = activation(linear, name=name + '_activation')
            return tf.reshape(linear, [-1, first_dim, second_dim])

        # v_t^i - The vectors to write to memory, for each write head `i`.
        write_vectors = _linear(self._num_writes, self._word_size, self.write_vector_linear, 'write_vectors')

        # e_t^i - Amount to erase the memory by before writing, for each write head.
        erase_vectors = _linear(self._num_writes, self._word_size, self.erase_vector_linear, 'erase_vectors',
                                tf.sigmoid)

        # f_t^j - Amount that the memory at the locations read from at the previous
        # time step can be declared unused, for each read head `j`.
        free_gate = tf.sigmoid(
            self.free_gate_linear(inputs))

        # g_t^{a, i} - Interpolation between writing to unallocated memory and
        # content-based lookup, for each write head `i`. Note: `a` is simply used to
        # identify this gate with allocation vs writing (as defined below).
        allocation_gate = tf.sigmoid(
            self.allocation_gate_linear(inputs))

        # g_t^{w, i} - Overall gating of write amount for each write head.
        write_gate = tf.sigmoid(
            self.write_gate_linear(inputs))

        # \pi_t^j - Mixing between "backwards" and "forwards" positions (for
        # each write head), and content-based lookup, for each read head.
        #num_read_modes = 1 + 2 * self._num_writes
        read_mode = tf.nn.softmax( _linear(self._num_reads, self.num_read_modes, self.read_mode_linear, name='read_mode'))
        # read_mode = snt.BatchApply(tf.nn.softmax)(
        #     _linear(self._num_reads, num_read_modes, name='read_mode'))

        # Parameters for the (read / write) "weights by content matching" modules.
        write_keys = _linear(self._num_writes, self._word_size, self.write_key_linear ,'write_keys')
        write_strengths = self.write_strengths_linear(
            inputs)

        read_keys = _linear(self._num_reads, self._word_size, self.read_keys_linear,'read_keys')
        read_strengths = self.read_strength_linear(inputs)

        result = {
            'read_content_keys': read_keys,
            'read_content_strengths': read_strengths,
            'write_content_keys': write_keys,
            'write_content_strengths': write_strengths,
            'write_vectors': write_vectors,
            'erase_vectors': erase_vectors,
            'free_gate': free_gate,
            'allocation_gate': allocation_gate,
            'write_gate': write_gate,
            'read_mode': read_mode,
        }
        return result 

    def _write_weights(self, inputs, memory, usage):
        """Calculates the memory locations to write to.
        This uses a combination of content-based lookup and finding an unused
        location in memory, for each write head.
        
        Args:

            inputs: Collection of inputs to the access module, including controls for
                how to chose memory writing, such as the content to look-up and the
                weighting between content-based and allocation-based addressing.
            memory: A tensor of shape  `[batch_size, memory_size, word_size]`
                containing the current memory contents.
            usage: Current memory usage, which is a tensor of shape `[batch_size,
                memory_size]`, used for allocation-based addressing.
        
        Returns:

            tensor of shape `[batch_size, num_writes, memory_size]` indicating where
                to write to (if anywhere) for each write head.
        """
        with tf.name_scope('write_weights'):
            # c_t^{w, i} - The content-based weights for each write head.
            write_content_weights = self._write_content_weight_mod(
                memory, inputs['write_content_keys'],
                inputs['write_content_strengths'])

            # a_t^i - The allocation weights for each write head.
            write_allocation_weights = self.write_allocation_weights(
                usage=usage,
                write_gates=(inputs['allocation_gate'] * inputs['write_gate']),
                num_writes=self._num_writes)

            # Expands gates over memory locations.
            allocation_gate = tf.expand_dims(inputs['allocation_gate'], -1)
            write_gate = tf.expand_dims(inputs['write_gate'], -1)

            # w_t^{w, i} - The write weightings for each write head.
            return write_gate * (allocation_gate * write_allocation_weights +
                                (1 - allocation_gate) * write_content_weights)
    
    def _read_weights(self, inputs, memory, prev_read_weights, link):
        """Calculates read weights for each read head.
        The read weights are a combination of following the link graphs in the
        forward or backward directions from the previous read position, and doing
        content-based lookup. The interpolation between these different modes is
        done by `inputs['read_mode']`.
        
        Args:

            inputs: Controls for this access module. This contains the content-based
                keys to lookup, and the weightings for the different read modes.
            memory: A tensor of shape `[batch_size, memory_size, word_size]`
                containing the current memory contents to do content-based lookup.
            prev_read_weights: A tensor of shape `[batch_size, num_reads,
                memory_size]` containing the previous read locations.
            link: A tensor of shape `[batch_size, num_writes, memory_size,
                memory_size]` containing the temporal write transition graphs.
        
        Returns:

            A tensor of shape `[batch_size, num_reads, memory_size]` containing the
            read weights for each read head.
        """
        with tf.name_scope('read_weights'):
            # c_t^{r, i} - The content weightings for each read head.
            content_weights = self._read_content_weight_mod(
                memory, inputs['read_content_keys'], inputs['read_content_strengths'])

            # Calculates f_t^i and b_t^i.
            forward_weights = self.directional_read_weights(
                link, prev_read_weights, forward=True)
            backward_weights = self.directional_read_weights(
                link, prev_read_weights, forward=False)

            backward_mode = inputs['read_mode'][:, :, :self._num_writes]
            forward_mode = (
                inputs['read_mode'][:, :, self._num_writes:2 * self._num_writes])
            content_mode = inputs['read_mode'][:, :, 2 * self._num_writes]

            read_weights = (
                tf.expand_dims(content_mode, 2) * content_weights + tf.reduce_sum(
                    tf.expand_dims(forward_mode, 3) * forward_weights, 2) +
                tf.reduce_sum(tf.expand_dims(backward_mode, 3) * backward_weights, 2))

            return read_weights


    def _memory_retention_vector(self, prev_read_weights, free_gate):
        free_gate = tf.expand_dims(free_gate, -1)
        free_read_weights = free_gate * prev_read_weights
        phi = reduce_prod(1 - free_read_weights, 1, name='phi')
        return phi

    def _usage_vector(self, prev_write_weight, prev_read_weight, prev_usage, free_gate) : 
        """Calculates the new memory usage u_t.
        Memory that was written to in the previous time step will have its usage
        increased; memory that was read from and the controller says can be "freed"
        will have its usage decreased.
        Args:
        write_weights: tensor of shape `[batch_size, num_writes,
            memory_size]` giving write weights at previous time step.
        free_gate: tensor of shape `[batch_size, num_reads]` which indicates
            which read heads read memory that can now be freed.
        read_weights: tensor of shape `[batch_size, num_reads,
            memory_size]` giving read weights at previous time step.
        prev_usage: tensor of shape `[batch_size, memory_size]` giving
            usage u_{t - 1} at the previous time step, with entries in range
            [0, 1].
        Returns:
        tensor of shape `[batch_size, memory_size]` representing updated memory
        usage.
        """
        # Calculation of usage is not differentiable with respect to write weights.
        prev_write_weight = tf.stop_gradient(prev_write_weight)
        usage = self._usage_after_write(prev_usage, prev_write_weight)
        usage = self._usage_after_read(usage, free_gate, prev_read_weight)
        return usage
 #       return prev_usage - prev_write_weight + (prev_usage * prev_write_weight) * self._memory_retention_vector(prev_read_weight, free_gate)

    def _usage_after_write(self, prev_usage, write_weights):
        """Calcualtes the new usage after writing to memory.
        Args:
        prev_usage: tensor of shape `[batch_size, memory_size]`.
        write_weights: tensor of shape `[batch_size, num_writes, memory_size]`.
        Returns:
        New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        with tf.name_scope('usage_after_write'):
            # Calculate the aggregated effect of all write heads
            write_weights = 1 - reduce_prod(1 - write_weights, 1)
            return prev_usage + (1 - prev_usage) * write_weights

    def _usage_after_read(self, prev_usage, free_gate, read_weights):
        """Calcualtes the new usage after reading and freeing from memory.
        Args:
        prev_usage: tensor of shape `[batch_size, memory_size]`.
        free_gate: tensor of shape `[batch_size, num_reads]` with entries in the
            range [0, 1] indicating the amount that locations read from can be
            freed.
        read_weights: tensor of shape `[batch_size, num_reads, memory_size]`.
        Returns:
        New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        with tf.name_scope('usage_after_read'):
            free_gate = tf.expand_dims(free_gate, -1)
            free_read_weights = free_gate * read_weights
            phi = reduce_prod(1 - free_read_weights, 1, name='phi')
            return prev_usage * phi




    def _allocation(self, usage):
        r"""Computes allocation by sorting `usage`.
        This corresponds to the value a = a_t[\phi_t[j]] in the paper.
        Args:

            usage: tensor of shape `[batch_size, memory_size]` indicating current
                memory usage. This is equal to u_t in the paper when we only have one
                write head, but for multiple write heads, one should update the usage
                while iterating through the write heads to take into account the
                allocation returned by this function.
        Returns:

            Tensor of shape `[batch_size, memory_size]` corresponding to allocation.
        """
        with tf.name_scope('allocation'):
            # Ensure values are not too small prior to cumprod.
            usage = _EPSILON + (1 - _EPSILON) * usage

            nonusage = 1 - usage
            sorted_nonusage, indices = tf.nn.top_k(
                nonusage, k=self.mem_size, name='sort')
            sorted_usage = 1 - sorted_nonusage
            prod_sorted_usage = tf.math.cumprod(sorted_usage, axis=1, exclusive=True)
            sorted_allocation = sorted_nonusage * prod_sorted_usage
            inverse_indices = batch_invert_permutation(indices)

            # This final line "unsorts" sorted_allocation, so that the indexing
            # corresponds to the original indexing of `usage`.
            return batch_gather(sorted_allocation, inverse_indices)






    def write_allocation_weights(self, usage, write_gates, num_writes):
        """Calculates freeness-based locations for writing to.
        This finds unused memory by ranking the memory locations by usage, for each
        write head. (For more than one write head, we use a "simulated new usage"
        which takes into account the fact that the previous write head will increase
        the usage in that area of the memory.)
       
        Args:

            usage: A tensor of shape `[batch_size, memory_size]` representing
                current memory usage.
            write_gates: A tensor of shape `[batch_size, num_writes]` with values in
                the range [0, 1] indicating how much each write head does writing
                based on the address returned here (and hence how much usage
                increases).
            num_writes: The number of write heads to calculate write weights for.
        
        Returns:

            tensor of shape `[batch_size, num_writes, memory_size]` containing the
                freeness-based write locations. Note that this isn't scaled by
                `write_gate`; this scaling must be applied externally.
        """
        with tf.name_scope('write_allocation_weights'):
            # expand gatings over memory locations
            write_gates = tf.expand_dims(write_gates, -1)

            allocation_weights = []
            for i in range(num_writes):
                allocation_weights.append(self._allocation(usage))
                # update usage to take into account writing to this new allocation
                usage += ((1 - usage) * write_gates[:, i, :] * allocation_weights[i])

            # Pack the allocation weights for the write heads into one tensor.
            return tf.stack(allocation_weights, axis=1)



    def _update_link_matrix(self, write_weights, prev_precedence_weights, prev_link) : 
        """Calculate the updated linkage state given the write weights.
        Args:
        write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
            containing the memory addresses of the different write heads.
        prev_state: `TemporalLinkageState` tuple containg a tensor `link` of
            shape `[batch_size, num_writes, memory_size, memory_size]`, and a
            tensor `precedence_weights` of shape `[batch_size, num_writes,
            memory_size]` containing the aggregated history of recent writes.
        Returns:
        A `TemporalLinkageState` tuple `next_state`, which contains the updated
        link and precedence weights.
        """
        link = self._link(prev_link, prev_precedence_weights,
                        write_weights)
        precedence_weights = self._precedence_weights(prev_precedence_weights,
                                                    write_weights)

        return link, precedence_weights




    def _precedence_weights(self, prev_precedence_weights, write_weights):
        """Calculates the new precedence weights given the current write weights.
        The precedence weights are the "aggregated write weights" for each write
        head, where write weights with sum close to zero will leave the precedence
        weights unchanged, but with sum close to one will replace the precedence
        weights.
        Args:
            prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
                memory_size]` containing the previous precedence weights.
            write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
                containing the new write weights.
        Returns:
            A tensor of shape `[batch_size, num_writes, memory_size]` containing the
            new precedence weights.
        """
        with tf.name_scope('precedence_weights'):
            write_sum = tf.reduce_sum(write_weights, 2, keepdims=True)
            return (1 - write_sum) * prev_precedence_weights + write_weights    

    def _link(self, prev_link, prev_precedence_weights, write_weights):
        """Calculates the new link graphs.
        For each write head, the link is a directed graph (represented by a matrix
        with entries in range [0, 1]) whose vertices are the memory locations, and
        an edge indicates temporal ordering of writes.
        
        Args:
            prev_link: A tensor of shape `[batch_size, num_writes, memory_size,
                memory_size]` representing the previous link graphs for each write
                head.
            prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
                memory_size]` which is the previous "aggregated" write weights for
                each write head.
            write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
                containing the new locations in memory written to.
        
        Returns:
            A tensor of shape `[batch_size, num_writes, memory_size, memory_size]`
            containing the new link graphs for each write head.
        """
        with tf.name_scope('link'):
            batch_size = tf.shape(prev_link)[0]
            write_weights_i = tf.expand_dims(write_weights, 3)
            write_weights_j = tf.expand_dims(write_weights, 2)
            prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights, 2)
            prev_link_scale = 1 - write_weights_i - write_weights_j
            new_link = write_weights_i * prev_precedence_weights_j
            link = prev_link_scale * prev_link + new_link
            # Return the link with the diagonal set to zero, to remove self-looping
            # edges.
            return tf.linalg.set_diag(
                link,
                tf.zeros(
                    [batch_size, self._num_writes, self.mem_size],
                    dtype=link.dtype))


    def directional_read_weights(self, link, prev_read_weights, forward):
        """Calculates the forward or the backward read weights.
        For each read head (at a given address), there are `num_writes` link graphs
        to follow. Thus this function computes a read address for each of the
        `num_reads * num_writes` pairs of read and write heads.
        Args:
        link: tensor of shape `[batch_size, num_writes, memory_size,
            memory_size]` representing the link graphs L_t.
        prev_read_weights: tensor of shape `[batch_size, num_reads,
            memory_size]` containing the previous read weights w_{t-1}^r.
        forward: Boolean indicating whether to follow the "future" direction in
            the link graph (True) or the "past" direction (False).
        Returns:
        tensor of shape `[batch_size, num_reads, num_writes, memory_size]`
        """
        with tf.name_scope('directional_read_weights'):
            # We calculate the forward and backward directions for each pair of
            # read and write heads; hence we need to tile the read weights and do a
            # sort of "outer product" to get this.
            expanded_read_weights = tf.stack([prev_read_weights] * self._num_writes,
                                            1)
            result = tf.matmul(expanded_read_weights, link, adjoint_b=forward)
            # Swap dimensions 1, 2 so order is [batch, reads, writes, memory]:
            return tf.transpose(result, perm=[0, 2, 1, 3])
