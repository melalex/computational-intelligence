from typing import Optional
import numpy as np
from tqdm.notebook import tqdm

from lib.model.neural_net import NeuralNetHistory, ValidationData
from lib.util.loss_function import LossFunction
from lib.util.member_function import MemberFunction
from lib.util.types import ArrayLike


class Anfis:
    __mem_fun: list[list[MemberFunction]]
    __consequent_params: ArrayLike
    __loss_fun: LossFunction
    __learning_rate: float

    def __init__(
        self,
        mem_fun: list[list[MemberFunction]],
        loss_fun: LossFunction,
        learning_rate: float,
    ) -> None:
        self.__mem_fun = mem_fun
        self.__loss_fun = loss_fun
        self.__learning_rate = learning_rate
        self.__consequent_params = np.random.uniform(
            low=0, high=1, size=(len(mem_fun), len(mem_fun) + 1)
        )

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        epochs: int = 100,
        validation_data: Optional[ValidationData] = None,
    ) -> NeuralNetHistory:
        train_loss_history = []
        val_loss_history = []

        with tqdm(total=epochs, desc="Progress") as pb:
            for _ in range(epochs):
                # Forward Pass: Compute outputs and perform LSE for consequent parameters
                y_hat, cache = self.forward(x, y)
                loss = self.__loss_fun.apply(y, y_hat)

                # Backward Pass: Gradient descent for antecedent parameters
                self.__backward(x, y, cache)

                pb_status = {"loss": loss}

                if validation_data is not None:
                    y_val_hat, _ = self.forward(validation_data.x)
                    val_loss = self.__loss_fun.apply(validation_data.y, y_val_hat)
                    val_loss_history.append(val_loss)

                    pb_status["val_loss"] = val_loss

                train_loss_history.append(loss)
                pb.update()
                pb.set_postfix(**pb_status)

        return NeuralNetHistory(train_loss_history, val_loss_history)

    def forward(self, x: ArrayLike, y: Optional[ArrayLike] = None):
        # Step 1: Compute membership values for each input and rule
        membership_values = np.array(
            [
                [mf.apply(x[i]) for mf in self.__mem_fun[i]]
                for i in range(len(self.__mem_fun))
            ]
        )

        # Step 2: Calculate rule firing strengths (Layer 2 in ANFIS)
        firing_strengths = np.prod(membership_values, axis=1)

        # Step 3: Normalize firing strengths (Layer 3 in ANFIS)
        firing_strengths_sum = np.sum(firing_strengths, axis=0)
        normalized_firing_strengths = firing_strengths / (firing_strengths_sum + 1e-9)

        # Step 4: Calculate rule outputs (Layer 4 in ANFIS)
        # Each rule has a linear consequent function: y_i = p_i * x_1 + q_i * x_2 + r_i
        if y is not None:
            # LSE Update for consequent parameters
            self.__consequent_params = self.__perform_lse(firing_strengths, x, y)

        rule_outputs = self.__calculate_consequent_layer(x)

        # Step 5: Aggregate the output using normalized firing strengths (Layer 5 in ANFIS)
        y_hat = np.sum(normalized_firing_strengths * rule_outputs, axis=0)

        return y_hat, (y_hat, rule_outputs, firing_strengths_sum, firing_strengths)

    def __backward(self, x, y, cache):
        y_hat, rule_outputs, firing_strengths_sum, firing_strengths = cache

        # Backward Step 1: Compute the error and its gradient
        d_loss = self.__loss_fun.apply_derivative(y, y_hat)

        # Backward Step 2: Calculate gradient
        for i in range(len(self.__mem_fun)):
            for j in range(len(self.__mem_fun[i])):
                mf = self.__mem_fun[i][j]
                x_i = x[i]

                # Derivative for mu_1 * mu_2. Commented because it is often equal to 0.
                # d_and = np.prod(
                #     [
                #         self.__mem_fun[i][k].apply(x_i)
                #         for k in range(len(self.__mem_fun[i]))
                #         if k != j
                #     ]
                # )
                d_w = (
                    d_loss
                    * rule_outputs[i]
                    * (firing_strengths_sum - firing_strengths[i])
                    / (firing_strengths_sum**2)
                    # * d_and
                )

                d_mean = d_w * mf.apply_d_mean(x_i)
                d_std = d_w * mf.apply_d_std(x_i)

                # Backward Step 3: Update params
                mf.set_mean(mf.get_mean() + self.__learning_rate * np.mean(d_mean))
                mf.set_std(mf.get_std() + self.__learning_rate * np.mean(d_std))

    def __perform_lse(self, firing_strengths, x, y):
        _, n_samples = x.shape

        # Construct the augmented input matrix (include a raw of 1's for the intercept term)
        x_aug = np.vstack((x, np.ones((1, n_samples))))

        return np.array(
            [np.linalg.lstsq((x_aug * w).T, y, rcond=None)[0] for w in firing_strengths]
        )

    def __calculate_consequent_layer(self, x):
        return np.dot(self.__consequent_params[:, :-1], x) + self.__consequent_params[
            :, -1
        ].reshape(-1, 1)
