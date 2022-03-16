####################################################
####             MAIN PARAMETERS                ####
####################################################
SimulateData  = True          # If False denotes training the CNN with SEGSaltData
ReUse         = False         # If False always re-train a network 
ReUse_best    = True          # ReUse the best trained model
velocity_flag = True         # If True denotes the process of radius inversion
DataDim       = [37,5001]    # Dimension of original one-shot seismic data
data_dsp_blk  = (1,2)         # Downsampling ratio of input
ModelDim      = [40,2500]     # Dimension of one velocity model
label_dsp_blk = (1,1)         # Downsampling ratio of output
position      = [6]           # The position of radius in filename [6,13,20]
Bubble_num    = 1             # N-bubble model
dh            = 10            # Space interval 


####################################################
####             NETWORK PARAMETERS             ####
####################################################
if SimulateData:
    Epochs        = 200       # Number of epoch
    TrainSize     = 500      # Number of training set
    TestSize      = 10       # Number of testing set
    TestBatchSize = 1
else:
    Epochs        = 50
    TrainSize     = 130      
    TestSize      = 10       
    TestBatchSize = 1
    
BatchSize         = 10        # Number of batch size
LearnRate         = 5e-3      # Learning rate
Nclasses          = 1         # Number of output channels
Inchannels        = 1        # Number of input channels, i.e. the number of shots
SaveEpoch         = 20        
DisplayStep       = 2         # Number of steps till outputting stats
