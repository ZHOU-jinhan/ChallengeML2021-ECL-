import LTSM as model_lstm
import CNN as model_cnn
import CNN_LSTM as model_crnn
import attention_CNN_LSTM as model_attcrnn
import GAN_CNN_LSTM as model_crgan

model=[model_lstm.myLSTM(),
       model_cnn.myCNN(),
       model_crnn.myCNN_LSTM(),
       model_attcrnn.myAttention_CNN_LSTM(),
       model_crgan.myCRGAN()]
i=0
print("===Start===")
for m in model:
    i+=1;print("\n\n===Model "+str(i)+"/"+str(len(model))+"===")
    m.train(plot=0)
    m.predict()
