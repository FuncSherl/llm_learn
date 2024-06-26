import os, logging

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

from transformer import decoder, encoder, transformer
# from datas.IWSLT_15_en2vi import dataloader
from datas.WMT_2014_en2de import dataloader
from configs import (
    EPOCHS,
    BATCHSIZE,
    DMODEL,
    ENCODER_NUM,
    DECODER_NUM,
    HEADNUM,
    DFF,
    WARMUP_STEPS,
    DROPOUT_PROB,
)
import numpy as np
import torch as pt
import math, time, datetime


class WMT2014EN2DE:
    modelname = "WMT2014EN2DE"

    def __init__(self, device="cpu") -> None:
        self.device = device
        self.train_dataloader = dataloader.get_train_dataloader(BATCHSIZE)
        self.test_dataloader = dataloader.get_test_dataloader(BATCHSIZE)
        self.dev_dataloader = dataloader.get_dev_dataloader(BATCHSIZE)

        self.src_dict_set, self.dst_dict_set = (
            dataloader.SRC_DICT_SET,
            dataloader.DST_DICT_SET,
        )

        self.src_word2token = dataloader.SRC_WORD2TOKEN
        self.dst_word2token = dataloader.DST_WORD2TOKEN

        self.src_token2word = dataloader.SRC_TOKEN2WORD
        self.dst_token2word = dataloader.DST_TOKEN2WORD
        self.src_special_tokens = [
            self.src_word2token[dataloader.PADSTR],
            self.src_word2token[dataloader.STARTSTR],
            self.src_word2token[dataloader.ENDSTR],
            self.src_word2token[dataloader.UNKSTR],
        ]
        self.dst_special_tokens = [
            self.dst_word2token[dataloader.PADSTR],
            self.dst_word2token[dataloader.STARTSTR],
            self.dst_word2token[dataloader.ENDSTR],
            self.dst_word2token[dataloader.UNKSTR],
        ]

        self.transformer_model = transformer.Transformer(
            DMODEL,
            ENCODER_NUM,
            DECODER_NUM,
            HEADNUM,
            DFF,
            len(self.src_dict_set),
            None if dataloader.USE_SAME_DICT else len(self.dst_dict_set),
            dataloader.MAX_SEQLEN_SRC,
            dataloader.MAX_SEQLEN_DST,
            self.src_special_tokens,
            self.dst_special_tokens,
            DROPOUT_PROB,
        ).to(self.device)

        self.batch_word2token = dataloader.batch_word2token
        self.batch_token2word = dataloader.batch_token2word

    def train(self, load_checkpoint_p=None):
        loss_func = pt.nn.CrossEntropyLoss(label_smoothing=0.1)  # 定义交叉熵损失函数
        # 定义优化器
        optimadam = pt.optim.Adam(
            self.transformer_model.parameters(), lr=2e-3, betas=(0.9, 0.98), eps=1e-9
        )

        # 定义学习率衰减策略
        def lr_strategy(step):
            step += 1  # incase step = -1
            return math.pow(DMODEL, -0.5) * min(
                math.pow(step, -0.5), step * math.pow(WARMUP_STEPS, -1.5)
            )

        sched = pt.optim.lr_scheduler.LambdaLR(optimadam, lr_strategy)

        # prepare path
        current_file_path = os.path.dirname(__file__)
        checkpoint_dirname = os.path.join(
            current_file_path, "%s_checkpoints" % (self.modelname)
        )
        os.makedirs(checkpoint_dirname, exist_ok=True)

        # args
        loss = 0
        show_gap = 20
        start_epoch = 0
        test_gap = 2000
        last_checkpint = None

        # load checkpoint if exists
        if load_checkpoint_p is not None:
            checkpoint = pt.load(load_checkpoint_p)
            self.transformer_model.load_state_dict(checkpoint["model_state_dict"])
            optimadam.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            sched.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        for epoch in range(start_epoch, EPOCHS):
            for stepcnt, (d, l) in enumerate(self.train_dataloader):
                self.transformer_model.train()
                st_time = time.time()
                optimadam.zero_grad()  # 梯度清零
                d = [x.split() for x in d]
                l = [x.split() for x in l]
                dat_src = self.batch_word2token(d, self.src_word2token)
                dat_dst = self.batch_word2token(l, self.dst_word2token)
                dat_src = pt.tensor(dat_src, dtype=pt.int32).to(self.device)
                dat_dst = pt.tensor(dat_dst, dtype=pt.int64).to(self.device)
                logit = self.transformer_model(dat_src, dat_dst[:, :-1])
                # flatten datas
                logit = logit.flatten(0, -2)
                dat_dst = dat_dst[:, 1:].flatten()

                loss = loss_func(logit, dat_dst)  # 计算损失
                loss.backward()  # 反向传播
                optimadam.step()  # 更新参数
                sched.step()
                ed_time = time.time()

                # show log
                if stepcnt % show_gap == 0:
                    logging.info(
                        "epoch [%d/%d] batch [%d/%d]  loss: %f    time: %f s/iter"
                        % (
                            epoch,
                            EPOCHS,
                            stepcnt,
                            len(self.train_dataloader),
                            loss,
                            (ed_time - st_time),
                        )
                    )
                # test
                if (stepcnt + 1) % test_gap == 0:
                    logging.info("Testing ...")
                    self.test(4)

            # model save per epoch
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.transformer_model.eval()
            checkpoint = {
                "model_state_dict": self.transformer_model.state_dict(),
                "optimizer_state_dict": optimadam.state_dict(),
                "epoch": epoch,
                "lr_scheduler_state_dict": sched.state_dict(),
            }
            modeln = "%s_%s_epoch_%d_loss_%.3f.pkl" % (
                self.modelname,
                timestamp,
                epoch,
                loss,
            )

            modeln = os.path.join(checkpoint_dirname, modeln)
            logging.info("epoch [%d/%d] Saving model to: %s" % (epoch, EPOCHS, modeln))
            pt.save(checkpoint, modeln)
            last_checkpint = modeln

    def test(self, testcnt=-1, load_checkpoint_p=None):
        # load checkpoint if exists
        if load_checkpoint_p is not None:
            checkpoint = pt.load(load_checkpoint_p)
            self.transformer_model.load_state_dict(checkpoint["model_state_dict"])

        self.transformer_model.eval()
        for stepcnt, (d, l) in enumerate(self.test_dataloader):
            if testcnt > 0 and stepcnt >= testcnt:
                break

            d = [x.split() for x in d]
            l = [x.split() for x in l]
            dat_src = self.batch_word2token(d, self.src_word2token)
            # dat_dst = self.batch_word2token(d, self.dst_word2token)
            dat_src = pt.tensor(dat_src, dtype=pt.int32).to(self.device)
            # dat_dst = pt.tensor(dat_dst, dtype=pt.int32).to(self.device)
            ret = self.transformer_model(dat_src, None)
            ret = self.batch_token2word(ret, self.dst_token2word)

            for i in range(len(d)):
                print(d[i], " --> ", l[i])
                print(ret[i], "\n")


if __name__ == "__main__":
    cuda_ava = pt.cuda.is_available()
    logging.info("cuda available: " + str(cuda_ava))
    if cuda_ava:
        logging.info("cuda device count: " + str(pt.cuda.device_count()))
        cur_ind = pt.cuda.current_device()
        logging.info(
            "current device index: %d  name: %s"
            % (cur_ind, pt.cuda.get_device_name(cur_ind))
        )

    device = pt.device("cuda" if pt.cuda.is_available else "cpu")
    wmtproc = WMT2014EN2DE(device=device)
    wmtproc.train()
    # wmtproc.test()
