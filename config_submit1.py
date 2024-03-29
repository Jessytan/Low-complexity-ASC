class config: 
    def __init__(self):

        #restore

        self.project_name = "Low-complexity-ASC"
        #
        self.dir_prob = 0.15
        self.dirs_path="/data/jessy/my/datasets/dirs"
        
        
        # wav to mel
        self.sample_rate = 44100 
        self.n_fft = 4096 
        self.win_length = self.n_fft
        self.hop_length =int(self.win_length/6) 
        self.n_mels = 256
        
        # when you use npdata which saved before
        self.reuse = True
        self.reusefolder = 'new/'#'fs44.1k_bin256_frame4096_hop0.25/'
        
        #when you train with all dataset
        self.include_val = False
        
        self.epochs = 100
        self.batch_size = 16
        self.num_workers = 8
        
        #augmentation
        self.DIFF_FREQ = True
        self.MIXUP = True
        self.SPEC_AUG = True
        self.lamda = 0.02#0.226
        
        # define knowledge distillation parameters
        self.temperature = 2.0
        
        
        
        
