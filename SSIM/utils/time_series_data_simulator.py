import numpy as np

class time_data_gen(object):
    """
    Generate stochastic time series data for time series imputation model
    
    Parameters:
    -----------
        
    mode: string
      sliding window mode, see 
      
        - GSW: 
            Generic sliding window algorithm. If the target_index is
            not the last few index, this mode is equivalent to Sliding
            window Algorithm with Future information (SWF).
        - VLSW: 
            Variable-length Sliding window Algorithm
            
    length: scalar or 1-D array
      The length of time series in sequence
      
        If a scalar is given, the mode must be 'GSW'
        If 1-D array is given, the mode must be 'VLSW'
      
    target_index: 1-D or 2-D array
      The index of target variables in the slided sequence
      
        If a 1-D array is given, the mode must be 'GSW'
        If a 2-D array is given, the mode must be 'VLSW'
    """
    
    def __init__(self,
                 steps_num = 5000,
                 mode = 'GSW',
                 length = 9,
                 target_index = [3, 4, 5]):
        self.steps_num = steps_num
        self.mode = mode
        self.length = length
        self.target_index = target_index

        
    def random_walking_gen(self, 
                  origin = [0, 0],
                  mean = [0, 0],
                  cov = [[1, 0], [0, 1]]):
        
        path = np.array([origin])
        steps = np.random.multivariate_normal(mean = mean,
                                 cov = cov,             
                                 size = self.steps_num)
        
        for step in steps:
            x = path[-1][0] + step[0]
            y = path[-1][1] + step[1]
            path = np.append(path, [[x, y]], axis = 0)
            
        return path
    
    def sliding(self, data):
        
        if self.mode == 'GSW':
            
            tmp_seq = np.copy(data[0: self.length])
            trg_data = np.expand_dims(tmp_seq, 
                                      axis = 0)[:, self.target_index, :]
            src_data = np.expand_dims(tmp_seq, 
                                      axis = 0)
            src_data[:, self.target_index, :] = 0
            len_before_data = np.array(self.target_index[0])
            
            for counter, value in enumerate(data[self.length:]):
                tmp_seq = np.copy(data[counter + 1:
                                counter + self.length + 1])
                trg_data = np.append(trg_data,
                                np.expand_dims(tmp_seq, 
                                   axis = 0)[:, self.target_index, :],
                                axis = 0)
                src_data = np.append(src_data,
                                np.expand_dims(tmp_seq, axis = 0),
                                axis = 0)
                src_data[-1, self.target_index, :] = 0
                len_before_data = np.append(len_before_data,
                                           self.target_index[0])
        else:
            print('incompleted!')
            
        
        return src_data, trg_data, len_before_data
        
        
    def data_prep(self, time_series_data = None):
        if time_series_data == None:
            time_series_data = self.random_walking_gen()
        
        return self.sliding(time_series_data)
