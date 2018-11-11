import math
import tensorflow as tf

def adv_cost_function(adv_py_x, py_x, Y):
    if FLAGS.adv_cost == 'ce':
        adv_cost = ce_loss(adv_py_x, Y)
    elif FLAGS.adv_cost == 'qe':
        adv_cost = qe_loss(adv_py_x, py_x)
    elif FLAGS.adv_cost == 'kl':
        adv_cost = kl_divergence_with_logit(adv_py_x, py_x)
    return adv_cost

# keepdims -> keepdims, ce_loss_sum, qe_loss_sum, accuracy, entropy_y
def rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def rampdown(epoch, rampdown_length, total_epoch):
    if epoch >= (total_epoch - rampdown_length):
        ep = (epoch - (total_epoch - rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / rampdown_length)
    else:
        return 1.0 

def batch_noise(shape, inner_seed, keep_prob):
    noise = tf.random_uniform(shape, seed=inner_seed, dtype=tf.float32)
    random_tensor = keep_prob + noise
    binary_tensor = tf.floor(random_tensor)
    return binary_tensor

def one_drop_noise(shape, inner_seed):
    noise = tf.random_uniform(shape, seed=inner_seed, dtype=tf.float32)
    max_val = tf.reduce_max(noise, axis=1, keep_dims=True)
    drop_points = tf.cast( tf.greater_equal(noise, max_val), tf.float32)
    binary_tensor = tf.ones_like(noise) - drop_points
    return binary_tensor

def adversarial_dropout(cur_mask, Jacobian, change_limit, name="ad"):
    
    dim = tf.reduce_prod(tf.shape(cur_mask)[1:])
    changed_mask = cur_mask
    
    if change_limit != 0 :
        
        dir = tf.reshape(Jacobian, [-1, dim])
        
        # mask (cur_mask=1->m=1), (cur_mask=0->m=-1)
        m = cur_mask
        m = 2.0*m - tf.ones_like(m)
        
        # sign of Jacobian  (J>0 -> s=1), (J<0 -> s= -1)
        s = tf.cast( tf.greater(dir, float(0.0)), tf.float32)
        s = 2.0*s - tf.ones_like(s)                  
        
        # remain (J>0, m=-1) and (J<0, m=1), which are candidates to be changed
        change_candidate = tf.cast( tf.less( s*m, float(0.0) ), tf.float32) # s = -1, m = 1
        ads_dL_dx = tf.abs(dir)
        
        # ordering abs_Jacobian for candidates
        # the maximum number of the changes is "change_limit"
        # draw top_k elements ( if the top k element is 0, the number of the changes is less than "change_limit" ) 
        left_values = change_candidate*ads_dL_dx
        with tf.device("/cpu:0"):
            min_left_values = tf.nn.top_k(left_values, change_limit)[0][:,-1]    
        change_target = tf.cast(  tf.greater(left_values, tf.expand_dims(min_left_values, -1) ), tf.float32)
        
        # changed mask with change_target
        changed_mask = (m - 2.0*m*change_target + tf.ones_like(m))*0.5 
    
    return changed_mask


def ce_loss(logit, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))

def qe_loss(logit1, logit2):
    logit1 = tf.nn.softmax(logit1)
    logit2 = tf.nn.softmax(logit2)
    return tf.reduce_mean( tf.squared_difference( logit1, logit2 ) )

  
def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm
  

def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp
