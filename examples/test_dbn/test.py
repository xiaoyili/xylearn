from heapq import heappush, nsmallest

from xylearn.hyper_optimizer import fmin, tpe, hp, STATUS_OK
from xylearn.hyper_optimizer.pyll import scope
from xylearn.models.dbn import kfold_train
from xylearn.utils.helper_util import drange2, set_precision
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
        def objfunc(pretrain_lr=0.1,
                    pretrain_epoch=10,
                    finetune_epoch=10,
                    finetune_lr=0.1,
                    isPCD=0,
                    dropout=0.2,
                    model_structure=[100, 10],
                    batch_size=20,
                    normalize_idx=0):
            # rounding them into integer
            batch_size = int(batch_size)

            loss = 0
            if not isinstance(pretrain_lr, dict):
                loss = kfold_train(pretrain_lr=set_precision(pretrain_lr),
                                   pretrain_epoch=pretrain_epoch,
                                   finetune_epoch=finetune_epoch,
                                   finetune_lr=set_precision(finetune_lr),
                                   isPCD=isPCD,
                                   dropout=dropout,
                                   model_structure=model_structure,
                                   batch_size=batch_size,
                                   normalize_idx=normalize_idx)

                # priority queue, insert sort according to loss
                params = {'pretrain_lr': set_precision(pretrain_lr),
                          'pretrain_epoch': pretrain_epoch,
                          'finetune_epoch': finetune_epoch,
                          'finetune_lr': set_precision(finetune_lr),
                          'isPCD': isPCD,
                          'dropout': dropout,
                          'batch_size': batch_size,
                          'model_structure': model_structure,
                          'normalize_idx': normalize_idx}

                heappush(self.perform_dict, [loss, params])

                print str(call_counter()) + '\t' + str(loss) + '\t' + str(params)

            return {'loss': loss, 'status': STATUS_OK}


        # define searching scope
        space = scope.objfunc(pretrain_lr=hp.lognormal('pretrain_lr', 0, 1),
                              finetune_lr=hp.lognormal('finetune_lr', 0, 1),
                              pretrain_epoch=hp.choice('pretrain_epoch', [10, 50, 100]),
                              finetune_epoch=hp.choice('finetune_epoch', [10, 50, 100]),
                              isPCD=hp.choice('isPCD', [0, 1]),
                              dropout=hp.choice('dropout', [0, 0.1]),
                              batch_size=hp.choice('batch_size', [10, 20, 40, 80, 100]),
                              model_structure=hp.choice('model_structure',
                                                        [[100, 10], [200, 20], [300, 30]]),
                              normalize_idx=hp.choice('normalize_idx', [0, 1, 2]))

        # begin search
        fmin(objfunc,
             space=space,
             algo=tpe.suggest,
             max_evals=100)

        # get top 5
        self.hyper_result = nsmallest(10, self.perform_dict)


if __name__ == "__main__":
    test = hyperOpt()
    test.begin()
    import cPickle

    print test.hyper_result

    f = open('/home/eric/Desktop/result.pkl', 'aw')
    cPickle.dump(test, f)



