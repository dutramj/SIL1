from abc import ABC, abstractmethod


class TVCBase(ABC):

    @abstractmethod
    def step(
        self,
        delta_q_cmd: float,
        delta_r_cmd: float,
        phase_id: int,
    ):
        pass
    