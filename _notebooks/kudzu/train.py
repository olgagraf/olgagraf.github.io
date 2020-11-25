import numpy as np
import tqdm

class Learner:
    
    def __init__(self, loss, model, opt, epochs):
        self.loss = loss
        self.model = model
        self.opt = opt
        self.epochs = epochs
        self.cbs = []
        
    def set_callbacks(self, cblist):
        for cb in cblist:
            self.cbs.append(cb)
            
    def __call__(self, cbname, *args):
        status = True
        for cb in self.cbs:
            cbwanted = getattr(cb, cbname, None)
            status = status and cbwanted and cbwanted(*args)
        return status
    
    def train_loop(self, dl, X_test, y_test):
        self.dl = dl # dl added in here
        bs = self.dl.bs
        datalen = len(self.dl.data)
        self.bpe = datalen//bs
        self.afrac = 0.
        if datalen % bs > 0:
            self.bpe  += 1
            self.afrac = (datalen % bs)/bs
        self('fit_start')
        for epoch in range(self.epochs):
            self('epoch_start', epoch)
            with tqdm.tqdm(total=self.bpe, desc='Batch', position=0,leave=True) as pbar:
                for inputs, targets in dl:
                    self("batch_start", dl.current_batch)

                    # make predictions
                    predicted = self.model(inputs)

                    # actual loss value
                    epochloss = self.loss(predicted, targets)
                    self('after_loss', epochloss)

                    # calculate gradient
                    intermed = self.loss.backward(predicted, targets)
                    self.model.backward(intermed)

                    # make step
                    self.opt.step(self.model) 

                    #training_accuracy value
                    probs = self.model(inputs)
                    predictions = 1*(probs >= 0.5)
                    training_acc = np.mean(targets == predictions)
                    self('add_acc', training_acc)
                    self('batch_end')
                    pbar.update()
            test_probs = self.model(X_test)
            test_predictions = 1*(test_probs >= 0.5)
            testing_acc = np.mean(y_test == test_predictions)
            self('add_test_acc', testing_acc)
            self('epoch_end')
        
        
        self('fit_end')
        return epochloss