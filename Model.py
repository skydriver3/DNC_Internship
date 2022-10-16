from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import Dataset
import pickle#5 as pickle 
import numpy as np 
import datetime
import time 


import dnc 

def masked_sigmoid_cross_entropy(mask,
                                 time_major=False, 
                                 time_average=False,
                                 log_prob_in_bits=False
                                 ):
  """Adds ops to graph which compute the (scalar) NLL of the target sequence.

  The logits parametrize independent bernoulli distributions per time-step and
  per batch element, and irrelevant time/batch elements are masked out by the
  mask tensor.

  Args:
    logits: `Tensor` of activations for which sigmoid(`logits`) gives the
        bernoulli parameter.
    target: time-major `Tensor` of target.
    mask: time-major `Tensor` to be multiplied elementwise with cost T x B cost
        masking out irrelevant time-steps.
    time_average: optionally average over the time dimension (sum by default).
    log_prob_in_bits: iff True express log-probabilities in bits (default nats).

  Returns:
    A `Tensor` representing the log-probability of the target.
  """
  t_m = 1 if not time_major else 0 
  b_m = 1 if time_major else 0 
  def loss_fn (y_true, y_pred) : 

    exp_mask = tf.expand_dims(mask, 0)


    xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    #assert not any(xent < 0), f'some values are negatives'
    #xent.shape.assert_has_rank(3)
    loss_time_batch = tf.reduce_sum(xent, axis=2, name="reducing_feature")
    masked_loss_time_batch = tf.multiply(loss_time_batch, exp_mask, name="applying_mask") 
    #assert masked_loss_time_batch.shape == mask.shape, f"shape not the same {masked_loss_time_batch.shape} vs {mask.shape}"
    loss_batch = tf.reduce_sum(masked_loss_time_batch , axis=t_m, name="reducing_time")
    #assert tuple(loss_batch.shape) == (None), f"the shape is not what we think it was {loss_batch.shape}"
    
    batch_size = tf.cast(tf.shape(y_true)[b_m], dtype=loss_time_batch.dtype)

    if time_average:
        mask_count = tf.reduce_sum(exp_mask, axis=t_m)
        loss_batch /= (mask_count + np.finfo(np.float32).eps)

    
    loss = tf.reduce_sum(loss_batch, name="reduce_batch") / batch_size
    if log_prob_in_bits:
        loss /= tf.log(2.)

    return loss

  return loss_fn

def getSplit(array, split) : 
  assert split <= 1, "split must be smaller than 1 "
  length = len(array) 
  return array[:int(length - (length * split))]

def run_model(model=True):
    """Runs model on input sequence."""
    # tf.debugging.experimental.enable_dump_debug_info(
    # dump_root="C:\\Users\\gaeta\\Documents\\Code\\DNC\\tmp\\tfdbg2_logdir",
    # tensor_debug_mode="FULL_HEALTH",
    # circular_buffer_size=1000)
    time_major = False
    validation_split =  0.2 
    batch_size = 32

    dataset = Dataset.Dataset()
    try : 
      fp = open("C:\\Users\\gaeta\\Documents\\Code\\DNC\\datasetV2.pickle", "rb")
      data = pickle.load(fp)
      dataset_tensors = Dataset.Dataset_tensors(*data)
    except FileNotFoundError: 
      fp = open("datasetV2.pickle", "wb")
      dataset_tensors = dataset.Create_list_cl(time_major)
      pickle.dump(dataset_tensors, fp)


    # with np.printoptions(threshold=np.inf) : 
    #   print(dataset_tensors.input[0, 140:150, :])
    #   print(dataset_tensors.output[0, 150:160, :])


    access_config = {
      "memory_size": 150,
      "word_size": 50,
      "num_reads": 4,
      "num_writes": 1,
    }
    controller_config = {
      "hidden_size": 64,
      "output_size": dataset.target_size
    }
    clip_value = 20


    inputs = tf.keras.Input(shape=(dataset.max_in_size + dataset.max_out_size,dataset.input_size), name="graph_query")
    outputs = None 
    if model : 
      dnc_core = dnc.DNC(lstm_units=128, 
                mem_size=access_config["memory_size"], 
                word_size=access_config["word_size"], 
                read_head_number=access_config["num_reads"], 
                write_head_number=access_config["num_writes"],
                clip_value=clip_value,
                dtype=tf.float32)
      print("\n\n TRAINING DNC \n\n")
      Rnn = tf.keras.layers.RNN(cell=dnc_core, time_major=time_major, return_sequences=True)
      outputs = Rnn(inputs) 
    else : 
      outputs = tf.keras.layers.LSTM(64, return_sequences=True, time_major=time_major) (inputs)
      outputs = tf.keras.layers.LSTM(128, return_sequences=True, time_major=time_major) (outputs)
      print("\n\n TRAINING LSTM \n\n")


    outputs = tf.keras.layers.Dense(64, activation="relu") (outputs) 
    outputs = tf.keras.layers.Dense(controller_config["output_size"])(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    mask_tensor = tf.constant(
      dataset_tensors.mask,
      dtype=tf.float32)
    

    log_dir = "C:\\Users\\gaeta\\Documents\\Code\\DNC\\tmp\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ("DNC" if model else "LSTM")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    model.compile(
    optimizer=tf.keras.optimizers.RMSprop(), 
    loss=masked_sigmoid_cross_entropy(mask_tensor, time_major=time_major), 
    metrics=["accuracy"])

    t_t = time.time()
    model.fit(
    dataset_tensors.input[:, :, :], 
    dataset_tensors.output[:, :, :], 
    batch_size=batch_size, 
    epochs=2, 
    validation_split=validation_split, 
    callbacks=[tensorboard_callback])
    t_t_res = time.time() - t_t 
    print(f"Train time : {t_t_res}s")

    t_a = time.time()
    res = model.predict(dataset_tensors.input[0:100, :, :])
    t_res = time.time() - t_a 
    print(f"Run time : {t_res}s")

    # with np.printoptions(threshold=np.inf) : 
    #   res = tf.round(tf.sigmoid(tf.constant(res))).numpy()
    #   print(res[0, 150:160, :])
    #   print(dataset_tensors.output[0, 150:160, :])

if __name__ == "__main__": 
  run_model(True) 
  run_model(False) 