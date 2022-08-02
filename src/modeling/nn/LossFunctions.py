import tensorflow as tf


class LossFunctions():
    """_summary_
    """
    def __init__(self) -> None:
        pass

    def wasserstein_loss(self, y_true, y_pred):
        return tf.math.reduce_mean(y_true*y_pred)

    def critic_loss(self, crit_model, real_pred, fake_pred,
                    real_spec, fake_spec):
        gp_weight = 0.1
        alpha = tf.random.normal([real_spec.shape[0], 1, 1], 0.0, 1.0)
        diff = fake_spec - real_spec
        interpolated = real_spec + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = crit_model(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        real_loss = self.wasserstein_loss(
            tf.ones_like(real_pred), real_pred)
        fake_loss = self.wasserstein_loss(
            -tf.ones_like(fake_pred), fake_pred)
        total_loss = real_loss + fake_loss + gp_weight*gp
        return total_loss

    def generator_loss(self, fake_pred):
        return self.wasserstein_loss(tf.ones_like(fake_pred), fake_pred)
