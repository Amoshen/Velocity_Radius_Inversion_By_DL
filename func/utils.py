import scipy.io


def SaveTrainResults_velocity(loss,loss_test,SavePath,bubble_num):
    data = {}
    data['loss_train'] = loss
    data['loss_test'] = loss_test
    scipy.io.savemat(SavePath+'TrainLoss_velocity_'+str(bubble_num),data)
    
def SaveTrainResults_radius(loss,loss_test,SavePath,bubble_num):
    data = {}
    data['loss_train'] = loss
    data['loss_test'] = loss_test
    scipy.io.savemat(SavePath+'TrainLoss_radius_'+str(bubble_num),data)

def SaveTestResults_velocity(Prediction,GT,radius,SavePath,bubble_num):
    data = {}
    data['GT']      = GT
    data['Prediction'] = Prediction
    data['Radius']     = radius
    scipy.io.savemat(SavePath+'TestResults_velocity_'+str(bubble_num),data) 

def SaveTestResults_radius(Prediction,GT,SavePath,bubble_num):
    data = {}
    data['GT']      = GT
    data['Prediction'] = Prediction
    scipy.io.savemat(SavePath+'TestResults_radius_'+str(bubble_num),data) 
    
    

