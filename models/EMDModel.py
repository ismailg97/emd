from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Callable, Union, List

import keras
import tensorflow as tf
from keras.metrics import Metric
from keras import Model
from keras.activations import linear, softmax
from keras.metrics import categorical_accuracy
from keras.optimizers import SGD, Optimizer, Adam
#from keras.optimizers.schedules import ExponentialDecay
from tensorflow import TensorShape

from loss_functions.emd import EmdWeightHeadStart, GroundDistanceManager, self_guided_earth_mover_distance
from models import operations


class EvaluationModel(ABC, Model):
    """Base class for the models to be evaluated."""

    _OPTIMIZER: ClassVar[Optimizer] = Adam
    _OPTIMIZER_MOMENTUM: ClassVar[float] = 0.98
    _METRICS: ClassVar[List[Metric]] = [
        categorical_accuracy,
        #one_off_accuracy
    ]
    _MODEL_NAME: ClassVar[str] = 'base_model'
    model: Model = None

    def __init__(
            self,
            number_of_classes: int,
            loss_function: Callable,
            learning_rate: float,
            ground_distance_path: Path,
            labels,
            **loss_function_kwargs,
    ):
        super(EvaluationModel, self).__init__()
        self.number_of_classes = number_of_classes
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.second_to_last_layer = None
        self.labels = labels

        self._build_model(
            number_of_classes=number_of_classes,
        )
        self._compile_model(
            ground_distance_path=ground_distance_path,
            loss_function=loss_function,
            **loss_function_kwargs
        )

    def call(self, inputs, **kwargs):
        return operations.build_general_model(input=inputs, nr_classes=self.number_of_classes)


    def compute_output_shape(self, input_shape):
        return TensorShape((
            input_shape[0],
            self.number_of_classes
        ))

    def _build_model(
            self,
            number_of_classes: int,
    ):
        input = tf.keras.Input(shape=(224, 224, 3))
        #output = self.call(self, inputs=input)
        output = operations.build_general_model(input=input, nr_classes=number_of_classes)
        self.model = keras.Model(input, output)

    def _compile_model(
            self,
            loss_function: Callable,
            ground_distance_path: Path,
            **loss_function_kwargs
    ):
        if loss_function == self_guided_earth_mover_distance:
            self.emd_weight_head_start = EmdWeightHeadStart()
            self.ground_distance_manager = GroundDistanceManager(ground_distance_path)
            self.ground_distance_manager.set_labels(self.labels)
        #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #    self.learning_rate,
        #    decay_steps=429,
        #    decay_rate=0.995
        #)
        self.compile(
            loss=loss_function(
                model=self,
                **loss_function_kwargs
            ),
            optimizer=self._OPTIMIZER(
                learning_rate=self.learning_rate,
                #nesterov=True,
                #momentum=self._OPTIMIZER_MOMENTUM
            ),
            #metrics=self._METRICS,
            run_eagerly=True
        )

    def callbacks(self, **kwargs):
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            mode="min"
        ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath="./checkpoints/classificatorEMD/{}/classificatorEMD.ckpt".format(self.number_of_classes),
                save_weights_only=False,
                mode="min",
                verbose=1,
                save_best_only=True,
                initial_value_threshold=0.075
                )
        ]
        if hasattr(self, 'ground_distance_manager'):
            labels = [batch[1] for batch in kwargs['x']]
            self.ground_distance_manager.set_labels(labels)
            callbacks.extend([self.emd_weight_head_start, self.ground_distance_manager])
        return callbacks
