class Result(object):
    def __init__(self, average_treatment_effect = None, standard_error = None, z = None,\
                 p_value = None, confidence_interval = None):
        self.average_treatment_effect = average_treatment_effect
        self.standard_error = standard_error
        self.z = z
        self.p_value = p_value
        self.confidence_interval = confidence_interval

    def show(self):
        print('*' * 20)
        print(f'average treatment effect: {self.average_treatment_effect}\n')
        print(f'standard error:           {self.standard_error:.3f}\n')
        print(f'p value:                  {self.p_value:.3f}\n')
        print(f'confidence interval:      {self.confidence_interval}')
