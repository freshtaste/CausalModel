class Result(object):
    def __init__(self, **kwargs):
        try:
            self.average_treatment_effect = kwargs.get('average_treatment_effect', None)
            self.standard_error = kwargs.get("standard_error", None)
            self.z = kwargs.get("z", None)
            self.p_value = kwargs.get('p_value', None)
            self.confidence_interval = kwargs.get('confidence_interval', None)
        except AttributeError:
            pass

    def show(self):
        print('*' * 20)
        print(f'average treatment effect: {self.average_treatment_effect}\n')
        print(f'standard error:           {self.standard_error}\n')
        print(f'p value:                  {self.p_value}\n')
        print(f'confidence interval:      {self.confidence_interval}')

var = Result(z="yilun", p_value=0.5)
var.show()