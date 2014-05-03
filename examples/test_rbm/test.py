from heapq import heappush, nlargest

from xylearn.hyper_optimizer import fmin, tpe, hp, STATUS_OK
from xylearn.hyper_optimizer.pyll import scope
from xylearn.models.rbm import kfold_train
# for debug
from xylearn.visualizer.terminal_printer import call_counter


class hyperOpt(object):
    """
    hyperOpt class
    """

    def __init__(self):
        self.perform_dict = []
        self.hyper_result = []


    def begin(self):
        @scope.define
        def objfunc(learning_rate=0.1,
                    batch_size=20,
                    n_hidden=50,
                    normalize_idx=0):
            # rounding them into integer
            batch_size = int(batch_size)

            loss = 0
            if not isinstance(learning_rate, dict):
                loss = kfold_train(learning_rate=learning_rate,
                                   batch_size=batch_size,
                                   n_hidden=n_hidden,
                                   normalize_idx=0,
                                   n_epochs=100)

                # priority queue, insert sort according to loss
                params = {'learning_rate': learning_rate,
                          'batch_size': batch_size,
                          'n_hidden': n_hidden,
                          'normalize_idx': normalize_idx}

                heappush(self.perform_dict, [loss, params])

                print str(call_counter()) + '\t' + str(loss) + '\t' + str(params)

            return {'loss': loss, 'status': STATUS_OK}


        # define searching scope
        space = scope.objfunc(learning_rate=hp.lognormal('learning_rate', 0, 1),
                              batch_size=hp.uniform('batch_size', 18, 25),
                              n_hidden=hp.choice('n_hidden', [5, 20, 40, 80, 100]),
                              normalize_idx=hp.choice('normalize_idx', [0, 1, 2]))

        # begin search
        fmin(objfunc,
             space=space,
             algo=tpe.suggest,
             max_evals=200)

        # get top 5
        self.hyper_result = nlargest(5, self.perform_dict)


if __name__ == "__main__":
    test = hyperOpt()
    test.begin()
    import cPickle

    print test.hyper_result

    f = open('/home/eric/Desktop/result.pkl', 'aw')
    cPickle.dump(test, f)

