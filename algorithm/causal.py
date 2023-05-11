import numpy as np
import tensorflow as tf
from utils.tf_utils import get_vars
from algorithm import basis_algorithm_collection

def get_init_for_SSL(shape, ar=False):
    if ar:
        t = 10.
    else:
        t = 1.
    v_0 = (t * np.ones(shape, np.float32)).tolist()
    v_1 = (-t * np.ones(shape, np.float32)).tolist()
    return tf.constant_initializer(np.concatenate([v_0, v_1], -1))
    
def sample_gumbel(shape, eps=1e-20):
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)


def gumbel_softmax(logits, temperature, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(input=logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature, axis=-1)
    # y = gumbel_softmax_sample / temperature

    if hard:
        k = tf.shape(input=logits)[-1] # 0.01 
        y_hard = tf.cast(tf.equal(y, tf.math.reduce_max(input_tensor=y, axis=-1, keepdims=True)),
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y


def CAUSAL(args):
    basis_alg_class = basis_algorithm_collection[args.basis_alg]
    class CausalReturnDecomposition(basis_alg_class):
        def __init__(self, args):


            super().__init__(args)

            self.train_info_r = {
                'R_loss': self.r_loss,

                'Gumble_Temperature': self.temperature
            }
            self.step_info.update({
                'ZR_filter_loss': self.causal_filter_B,

                'AR_filter_loss': self.causal_filter_C,
            })
            if args.causal_bias_correction:
                self.train_info_r['R_var'] = self.r_var
            if self.args.policy_learning_with_causal:
                self.train_info_r.update({
                    'Dyn_loss': self.dyn_loss,
                    'Dyn_prob': self.dyn_prob,
                    'Dyn/mean': self.mean,
                    'Dyn/mean_label': self.mean_label,
                    # 'Dyn/logmix': self.logmix,
                    # 'Dyn/logstd': self.logstd,
                })
                self.step_info.update({
                    'ZZ_filter_loss': self.causal_filter_D,
                    'ZZ_filter_loss_aux': self.causal_filter_D_aux,
                    'AZ_filter_loss': self.causal_filter_E,
                })
            self.train_info_q = {**self.train_info_q, **self.train_info_r}
            self.train_info = {**self.train_info, **self.train_info_r}
            self.temperature_value = self.args.temperature
        
        def create_inputs(self):
            super().create_inputs()

            self.causal_raw_obs_ph_for_policy = tf.compat.v1.placeholder(tf.float32, [None]+self.args.obs_dims)
            self.causal_raw_obs_next_ph_for_policy = tf.compat.v1.placeholder(tf.float32, [None]+self.args.obs_dims)
            
            self.causal_raw_obs_ph = tf.compat.v1.placeholder(tf.float32, [None, None]+self.args.obs_dims)
            self.causal_raw_obs_next_ph = tf.compat.v1.placeholder(tf.float32, [None, None]+self.args.obs_dims)
            self.causal_acts_ph = tf.compat.v1.placeholder(tf.float32, [None, None]+self.args.acts_dims)
            self.causal_rews_ph = tf.compat.v1.placeholder(tf.float32, [None, 1])
            self.causal_done_ph = tf.compat.v1.placeholder(tf.float32, [None, None, 1])
            if self.args.causal_bias_correction:
                self.causal_var_coef_ph = tf.compat.v1.placeholder(tf.float32, [None, 1])
            self.temperature = tf.compat.v1.placeholder(tf.float32, [])

        def create_normalizer(self):
            super().create_normalizer()

            if self.args.obs_normalization:
                self.causal_obs_ph_for_policy = self.obs_normalizer.normalize(self.causal_raw_obs_ph_for_policy)
                self.causal_obs_next_ph_for_policy = self.obs_normalizer.normalize(self.causal_raw_obs_next_ph_for_policy)
                self.causal_obs_ph = self.obs_normalizer.normalize(self.causal_raw_obs_ph)
                self.causal_obs_next_ph = self.obs_normalizer.normalize(self.causal_raw_obs_next_ph)
            else:
                self.causal_obs_ph_for_policy = self.causal_raw_obs_ph_for_policy
                self.causal_obs_next_ph_for_policy = self.causal_raw_obs_next_ph_for_policy
                self.causal_obs_ph = self.causal_raw_obs_ph
                self.causal_obs_next_ph = self.causal_raw_obs_next_ph

        def create_network(self):
            def mlp_causal(causal_obs_ph, causal_acts_ph):
                causal_state_ph = tf.concat([causal_obs_ph, causal_acts_ph], axis=-1)
                with tf.compat.v1.variable_scope('net', initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")):
                    r_dense1 = tf.compat.v1.layers.dense(causal_state_ph, 256, activation=tf.nn.relu, name='r_dense1')
                    r_dense2 = tf.compat.v1.layers.dense(r_dense1, 256, activation=tf.nn.relu, name='r_dense2')
                    r = tf.compat.v1.layers.dense(r_dense2, 1, name='r')
                return r

            if not self.args.full_structure_initial:
                with tf.compat.v1.variable_scope('causal', initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")):
                    self.SSL_B = tf.compat.v1.get_variable('B', self.args.state_dim + [2], tf.float32)
                    self.SSL_C = tf.compat.v1.get_variable('C', self.args.acts_dims + [2], tf.float32)
                if self.args.policy_learning_with_causal:
                    with tf.compat.v1.variable_scope('dyn', initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")):
                        self.SSL_D = tf.compat.v1.get_variable('D', self.args.state_dim + self.args.state_dim + [2], tf.float32)
                        self.SSL_E = tf.compat.v1.get_variable('E', self.args.acts_dims + self.args.state_dim + [2], tf.float32)    
           
            else:
                with tf.compat.v1.variable_scope('causal'):
                    self.SSL_B = tf.compat.v1.get_variable('B', self.args.state_dim + [2], tf.float32, initializer=get_init_for_SSL(self.args.state_dim +[1]))
                    self.SSL_C = tf.compat.v1.get_variable('C', self.args.acts_dims + [2], tf.float32, initializer=get_init_for_SSL(self.args.acts_dims +[1]))
                if self.args.policy_learning_with_causal:
                    with tf.compat.v1.variable_scope('dyn'):
                        self.SSL_D = tf.compat.v1.get_variable('D', self.args.state_dim + self.args.state_dim + [2], tf.float32, initializer=get_init_for_SSL(self.args.state_dim + self.args.state_dim +[1]))
                        self.SSL_E = tf.compat.v1.get_variable('E', self.args.acts_dims + self.args.state_dim + [2], tf.float32, initializer=get_init_for_SSL(self.args.acts_dims + self.args.state_dim +[1]))  
            
            reward_net = mlp_causal if len(self.args.obs_dims)==1 else conv_causal
            # =========================== configure masks =========================== #
            self.SSL_B_hard = gumbel_softmax(self.SSL_B, self.temperature, hard=True)[:, :1]
            self.SSL_C_hard = gumbel_softmax(self.SSL_C, self.temperature, hard=True)[:, :1]
            if self.args.policy_learning_with_causal:
                self.SSL_D_hard = gumbel_softmax(self.SSL_D, self.temperature, hard=True)[:, :, 0]
                self.SSL_E_hard = gumbel_softmax(self.SSL_E, self.temperature, hard=True)[:, :, 0]

            # ===================== reward for model estimation ===================== #
            ssl_zr =  tf.multiply(self.causal_obs_ph,  self.SSL_B_hard[:, 0])
            ssl_ar =  tf.multiply(self.causal_acts_ph,  self.SSL_C_hard[:, 0])
            with tf.compat.v1.variable_scope('causal'):
                self.causal_rews_pred = reward_net(ssl_zr, ssl_ar)
                if self.args.apply_accurate_loss:
                    self.causal = tf.reduce_sum(input_tensor=self.causal_rews_pred* (1-self.causal_done_ph), axis=1) \
                        / tf.reduce_sum(1 - self.causal_done_ph, axis=1)
                
                else:
                    self.causal = tf.reduce_mean(input_tensor=self.causal_rews_pred, axis=1)
            # ===================== reward for policy learning ===================== #


            state_mask = tf.cast(tf.math.greater(self.SSL_B[:, 0], self.SSL_B[:, 1]), dtype=tf.float32)
            ssl_zr_for_policy_learning =  tf.multiply(self.causal_obs_ph_for_policy, state_mask)
            action_mask = tf.cast(tf.math.greater(self.SSL_C[:, 0], self.SSL_C[:, 1]), dtype=tf.float32)
            ssl_ar_for_policy_learning =  tf.multiply(self.acts_ph,  action_mask)
            self.causal_filter_B = tf.reduce_mean(state_mask)
            self.causal_filter_C = tf.reduce_mean(action_mask)
            with tf.compat.v1.variable_scope('causal', reuse=True):
                self.rews_ph = self.rews_pred = tf.stop_gradient(
                    reward_net(ssl_zr_for_policy_learning, ssl_ar_for_policy_learning))
            
            # ===================== dynamics for policy learning ================= #
            if self.args.policy_learning_with_causal:
                def dyn_causal(causal_obs_ph, causal_acts_ph, num_mix = 3): # state_dim, batch_size, state_dim + action_dim
                    causal_state_ph = tf.concat([causal_obs_ph, causal_acts_ph], axis=-1)
                    with tf.compat.v1.variable_scope('dyn', initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")):
                        dyn_dense1 = tf.compat.v1.layers.dense(causal_state_ph, 256, activation=tf.nn.relu, name='dyn_dense1')
                        dyn_dense2 = tf.compat.v1.layers.dense(dyn_dense1, 256, activation=tf.nn.relu, name='dyn_dense2')
                        output = tf.compat.v1.layers.dense(dyn_dense2, 1 * num_mix * 3, name='dyn')

                        # output_flat = tf.reshape(output, [-1, 1 * num_mix * 3])
                        return self.get_mdn_coef(output)
                ssl_zz = tf.stack([tf.multiply(self.causal_obs_ph_for_policy,  self.SSL_D_hard[:, i]) for i in range(self.args.state_dim[0])])
                ssl_az = tf.stack([tf.multiply(self.acts_ph,  self.SSL_E_hard[:, i]) for i in range(self.args.state_dim[0])])
                
                z_logmix, z_mean, z_logstd = dyn_causal(ssl_zz, ssl_az)
                next_obs_for_dyn = tf.expand_dims(tf.transpose(self.causal_obs_next_ph_for_policy, [1, 0]), -1)
                self.dyn_loss, self.dyn_prob = self.get_lossfunc(z_logmix, z_mean, z_logstd, next_obs_for_dyn)
  
            # ===================== input for policy learning ===================== #
            if self.args.policy_learning_with_causal:

                self.zz_label = tf.concat([tf.zeros(self.args.state_dim + self.args.state_dim + [1]), \
                        tf.ones(self.args.state_dim + self.args.state_dim + [1])], -1)
                self.az_label =  tf.concat([tf.zeros(self.args.acts_dims + self.args.state_dim + [1]), \
                        tf.ones(self.args.acts_dims + self.args.state_dim + [1])], -1)
                self.filter_mask = tf.compat.v1.one_hot(tf.range(self.args.state_dim[0]), self.args.state_dim[0])

                state_dim_in_sr = tf.cast(tf.math.greater(self.SSL_B[:, 0], self.SSL_B[:, 1]), dtype=tf.float32)
                state_dim_in_ss = tf.cast(tf.math.greater(self.SSL_D[:, :, 0], self.SSL_D[:, :, 1]), dtype=tf.float32)
                self.causal_filter_D = tf.reduce_sum(state_dim_in_ss * (1 - self.filter_mask)) / tf.reduce_sum(1 - self.filter_mask)
                self.causal_filter_D_aux = tf.reduce_sum(state_dim_in_ss * self.filter_mask) / tf.reduce_sum(self.filter_mask)
                self.causal_filter_E = tf.reduce_mean(tf.cast(tf.math.greater(self.SSL_E[:, :, 0], self.SSL_E[:, :, 1]), dtype=tf.float32))
                state_dim_in_ss = state_dim_in_ss * tf.tile(tf.expand_dims(state_dim_in_sr, 0), [self.args.state_dim[0], 1])
                state_dim_in_ss = tf.reduce_max(state_dim_in_ss, 0)
                state_to_policy = tf.reduce_max(tf.concat([state_dim_in_ss, state_dim_in_sr], -1), -1)
                self.obs_ph = tf.multiply(self.causal_obs_ph_for_policy,  state_to_policy)
                self.obs_next_ph = tf.multiply(self.causal_obs_next_ph_for_policy,  state_to_policy)

            else:
                self.obs_ph = self.causal_obs_ph_for_policy
                self.obs_next_ph = self.causal_obs_next_ph_for_policy


            super().create_network()
            
        def create_operators(self):
            super().create_operators() 

            self.r_loss = tf.reduce_mean(input_tensor=tf.square(self.causal-self.causal_rews_ph))
            if self.args.causal_bias_correction:
                assert self.args.causal_sample_size>1
                n = self.args.causal_sample_size
                self.r_var_single = tf.reduce_sum(input_tensor=tf.square(self.causal_rews_pred-tf.reduce_mean(input_tensor=self.causal_rews_pred, axis=1, keepdims=True)), axis=1) / (n-1)
                self.r_var = tf.reduce_mean(input_tensor=self.r_var_single*self.causal_var_coef_ph/n)
                self.r_total_loss = self.r_loss - self.r_var
            else:
                self.r_total_loss = self.r_loss


            if self.args.sparsity_loss_type == "l1":
                raise NotImplementedError
            elif self.args.sparsity_loss_type == "cross_entropy":
                zr_label = tf.concat([tf.zeros(self.args.state_dim + [1]), tf.ones(self.args.state_dim + [1])], -1)
                ar_label =  tf.concat([tf.zeros(self.args.acts_dims + [1]), tf.ones(self.args.acts_dims + [1])], -1)
                self.causal_filter_loss = \
                        self.args.zr_sparsity_coef * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.SSL_B, labels=zr_label)) + \
                        self.args.ar_sparsity_coef *  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.SSL_C, labels=ar_label))
            if self.args.causal_discovery_type == "reduce_sparsity":
                self.r_total_loss += tf.cast(tf.cond(tf.less(self.args.reward_loss_sh, tf.stop_gradient(self.r_loss)), lambda: -1, lambda: 1), tf.float32) * self.causal_filter_loss
            elif self.args.causal_discovery_type == "none":
                self.r_total_loss += tf.cast(tf.cond(tf.less(self.args.reward_loss_sh, tf.stop_gradient(self.r_loss)), lambda: 0, lambda: 1), tf.float32) * self.causal_filter_loss
            elif self.args.causal_discovery_type == "increase_sparsity":
                self.r_total_loss += self.causal_filter_loss
            
            self.r_optimizer = tf.compat.v1.train.AdamOptimizer(self.args.r_lr)
            self.r_train_op = self.r_optimizer.minimize(self.r_total_loss, var_list=get_vars('causal/'))
            self.q_train_op = tf.group([self.q_train_op, self.r_train_op])            
            if self.args.policy_learning_with_causal:

                if self.args.sparsity_loss_type == "l1":
                    raise NotImplementedError
                elif self.args.sparsity_loss_type == "cross_entropy":
                    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.SSL_D, labels=self.zz_label)
                    self.cross_entropy_loss = cross_entropy_loss
                    self.dyn_filter_loss = \
                        self.args.zz_sparsity_coef * tf.reduce_sum(cross_entropy_loss* (1-self.filter_mask)) / tf.reduce_sum(1-self.filter_mask) + \
                        self.args.zz_sparsity_coef_aux * tf.reduce_sum(cross_entropy_loss * self.filter_mask) / tf.reduce_sum(self.filter_mask) + \
                        self.args.az_sparsity_coef *  \
                            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.SSL_E, labels=self.az_label))
                self.dyn_total_loss = self.dyn_loss
                if self.args.causal_discovery_type == "reduce_sparsity":
                    self.dyn_total_loss += tf.cast(tf.cond(tf.less(self.args.dyn_loss_sh, tf.stop_gradient(self.dyn_loss)), lambda: -1, lambda: 1), tf.float32) * self.dyn_filter_loss
                elif self.args.causal_discovery_type == "none":
                    self.dyn_total_loss += tf.cast(tf.cond(tf.less(self.args.dyn_loss_sh, tf.stop_gradient(self.dyn_loss)), lambda: 0, lambda: 1), tf.float32) * self.dyn_filter_loss
                elif self.args.causal_discovery_type == "increase_sparsity":
                    self.dyn_total_loss += self.dyn_filter_loss
                
                self.dyn_optimizer = tf.compat.v1.train.AdamOptimizer(self.args.r_lr)
                self.dyn_train_op = self.r_optimizer.minimize(self.dyn_total_loss, var_list=get_vars('dyn/'))
                self.q_train_op = tf.group([self.q_train_op, self.dyn_train_op])

            self.init_op = tf.compat.v1.global_variables_initializer()
        def tf_lognormal(self, y, mean, logstd):
            logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
            return -0.5 * ((y - mean) / logstd) ** 2 - tf.math.log(logstd) - logSqrtTwoPI
        def get_mdn_coef(self, output):
            logmix, mean, logstd = tf.split(output, 3, -1)
            logmix = logmix - tf.reduce_logsumexp(input_tensor=logmix, axis=-1, keepdims=True)
            # self.logmix, self.mean, self.logstd = tf.reduce_mean(tf.reduce_sum(tf.exp(logmix), -1)), tf.reduce_mean(mean), tf.reduce_mean(logstd)
            return logmix, mean, logstd
            
        def get_lossfunc(self, logmix, mean, logstd, y):
            # logstd = 1.
            v = logmix + self.tf_lognormal(y, mean, logstd)
            self.mean = tf.reduce_mean(tf.reduce_sum(tf.exp(logmix) * mean, -1, keepdims=True))
            self.mean_label = tf.reduce_mean(y)
            v = tf.reduce_logsumexp(v, -1, keepdims=True)
            return -tf.reduce_mean(v), tf.reduce_mean(tf.exp(v))

        def get_reward(self, obs, obs_next, act):
            return self.sess.run(self.rews_ph, feed_dict={
                self.causal_obs_ph_for_policy: [obs], 
                self.causal_obs_next_ph_for_policy:[obs_next],
                self.acts_ph: [act]})
        
        def feed_dict(self, batch):
            batch_size = np.array(batch['obs']).shape[0]
            basis_feed_dict = super().feed_dict(batch)
            del basis_feed_dict[self.rews_ph]
            del basis_feed_dict[self.raw_obs_ph]
            del basis_feed_dict[self.raw_obs_next_ph]
            def one_hot(idx):
                idx = np.array(idx)
                batch_size, sample_size = idx.shape[0], idx.shape[1]
                idx = np.reshape(idx, [batch_size*sample_size])
                res = np.zeros((batch_size*sample_size, self.acts_num), dtype=np.float32)
                res[np.arange(batch_size*sample_size),idx] = 1.0
                res = np.reshape(res, [batch_size, sample_size, self.acts_num])
                return res
            causal_feed_dict = {
                **basis_feed_dict, **{
                    self.causal_raw_obs_ph: batch['causal_obs'],
                    self.causal_raw_obs_next_ph: batch['causal_obs_next'],
                    self.causal_acts_ph: batch['causal_acts'] if self.args.env_category!='atari' else one_hot(batch['causal_acts']),
                    self.causal_rews_ph: batch['causal_rews'],
                    self.causal_raw_obs_ph_for_policy: batch['obs'],
                    self.causal_raw_obs_next_ph_for_policy: batch['obs_next'],
                    self.causal_done_ph: batch['causal_done'],
                    self.temperature: self.temperature_value,
                } 
            }

            if self.args.causal_bias_correction:
                causal_feed_dict[self.causal_var_coef_ph] = batch['causal_var_coef']
            return causal_feed_dict
        def step(self, obs, explore=False, test_info=False):
            if explore:
                noise = np.random.normal(0.0, 1.0, size=self.args.acts_dims)
            else:
                noise = np.zeros(shape=self.args.acts_dims, dtype=np.float32)
            feed_dict = {
                self.causal_raw_obs_ph_for_policy: [obs],
                self.pi_noise_ph: [noise]
            }
            
            action, info = self.sess.run([self.pi_act, self.step_info], feed_dict)
            if np.isnan(action).any():
                import ipdb; ipdb.set_trace()
            action = action[0]

            if test_info: return action, info
            return action
        def get_structure(self, ):
            if self.args.policy_learning_with_causal:
                B, C, D, E = self.sess.run([self.SSL_B, self.SSL_C, self.SSL_D, self.SSL_E])
                causal_structure = {
                    "B":B,
                    "C": C,
                    "D":D,
                    "E":E,
                }
            else:
                B, C = self.sess.run([self.SSL_B, self.SSL_C])
                causal_structure = {
                    "B":B,
                    "C": C,
                }
            return causal_structure
        def adjust_temperature(self, ):
            self.temperature_value = max(self.temperature_value * 0.9995, 0.05)  
            # print("update temperature: ", self.temperature_value)
        def train(self, batch):
            feed_dict = self.feed_dict(batch)
            info, _, _ = self.sess.run([self.train_info, self.pi_train_op, self.q_train_op], feed_dict)
            return info

    return CausalReturnDecomposition(args)
