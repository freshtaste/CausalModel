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
        
    
    def __str__(self):
        output = ('*' * 20 + '\n' 
            + f'average treatment effect: {self.average_treatment_effect}\n' 
            + '\n' 
            + f'standard error:           {self.standard_error}\n' 
            + '\n' 
            + f'p value:                  {self.p_value}\n' 
            + '\n' 
            + f'confidence interval:      {self.confidence_interval}')
        return output
        

    def show(self):
        print(self.__str__())
