
from modular.component import FCNetwork, ImageNetwork, LayeredNetwork, Concatenation, Addition, Loss, EmptyComponent
from modular.tensor_cloud import input_tensor, TensorCloud

from functools import lru_cache
import numpy as np
import tensorflow as tf

COLORS = "red", "green", "yellow", "blue"
COLOR_DICTIONARY = {
    "red" : [255, 0, 0],
    "green" : [0, 255, 0],
    "yellow" : [255, 255, 0],
    "blue" : [0, 0, 255]
}
DIMENSIONS = 3

def construct_network(hidden_size, number_layers, conv_size, n_conv_layers, image_width, image_height, filter_size=4):
    with tf.variable_scope("constant_colors"):
        constant_by_color = TensorCloud({
            color : tf.add(tf.constant(np.zeros((1, image_width, image_height, 3), dtype=np.float32)),
                           tf.constant(np.array(COLOR_DICTIONARY[color], dtype=np.float32)),
                           name=f"image_{color}")
                for color in COLORS
        })
    hidden_layers = [hidden_size] * number_layers
    state_to_features = FCNetwork("state_to_features", len(COLORS) * DIMENSIONS, hidden_layers, hidden_size)
    image_to_features = ImageNetwork("image_to_features",
                                    filter_size=filter_size,
                                    in_width=image_width, in_height=image_height,
                                    channel_sizes=[conv_size] * n_conv_layers, output_channels=conv_size, stride=1,
                                    hidden_layers=hidden_layers, hidden_size=hidden_size)
    features_to_color = FCNetwork("features_to_color", hidden_size * 2, hidden_layers, hidden_size)
    reacher_to_protocol = FCNetwork("reacher_to_protocol", hidden_size, hidden_layers, hidden_size)
    pusher_to_protocol = FCNetwork("blockpush_to_protocol", 2 * hidden_size, hidden_layers, hidden_size)
    protocol_to_linker = lru_cache(None)(lambda links: FCNetwork(
            f"protocol_to_{links}linker", hidden_size + 2 * (links + 1), hidden_layers, 1 + links)
    )
    class ReshapeIntoImage(EmptyComponent):
        def __call__(self, arg):
            return tf.reshape(arg, [-1, image_width, image_height, 3])
    protocol_to_end_delta = LayeredNetwork("protocol_to_end_delta",
                                           lambda: FCNetwork("protocol_to_end_delta", hidden_size, hidden_layers, image_width * image_height * 3),
                                           ReshapeIntoImage)
    state = TensorCloud.input(len(COLORS) * DIMENSIONS, "block_locations")
    joint_angles = TensorCloud({links : input_tensor(2 * (links + 1), "joint_angles_%s" % links) for links in (3, 4, 5)})
    images = TensorCloud.input([image_width, image_height, 3], "images")
    constant_features = constant_by_color | image_to_features
    features = (images | image_to_features).label("state") + (state | state_to_features).label("images")
    color_locs = TensorCloud.product(features, constant_features, Concatenation("features_with_image_label")) | features_to_color
    reach_protocol = color_locs | reacher_to_protocol
    push_protocol = TensorCloud.product(color_locs, color_locs, Concatenation("concat_push"), lambda x, y: x < y) | pusher_to_protocol
    protocol = push_protocol + reach_protocol
    robot_input = TensorCloud.product(protocol, joint_angles, Concatenation("concat_protocol_angles"))
    output = robot_input | (lambda x: protocol_to_linker(x[-1]))
    end_image = TensorCloud.product(protocol | protocol_to_end_delta, images, Addition("end_image"))
    return (images, state, joint_angles), (output, end_image)
