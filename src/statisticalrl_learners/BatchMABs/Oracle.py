from statisticalrl_learners.BatchMABs import BatchBanditAgent


class BatchOracle(BatchBanditAgent):
    """Oracle"""
    def __init__(self,env):
        self.env=env
        nA = env.action_space.n
        BatchBanditAgent.__init__(self, nA, name="Oracle")
        self.policy = [self.env.bestarm]

    def name(self):
        return "Oracle"

    # def reset(self):
    #     ()

    def play(self):
        return self.env.bestarm