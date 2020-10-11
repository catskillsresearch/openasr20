import os
from tqdm.notebook import tqdm
import logging, torch, time, sys
from utils import constant
from torch.autograd import Variable
from utils.functions import save_model
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer, calculate_cer_en_zh
from torch.cuda.amp import GradScaler, autocast # Tensor Cores

class TrainerVanilla():
    """
    Trainer class
    """
    def __init__(self):
        logging.info("Trainer is initialized")
        self.writer = SummaryWriter()

    def train(self, model, train_loader, train_sampler, opt, loss_type,
              start_epoch, num_epochs, label2id, id2label, just_once = False):
        """
        Training
        args:
            model: Model object
            train_loader: DataLoader object of the training set
            opt: Optimizer object
            start_epoch: start epoch (> 0 if you resume the process)
            num_epochs: last epoch
        """
        print("WELCOME TO PAN AMERICAN AIRWAYS")
        history = []
        start_time = time.time()
        smoothing = constant.args.label_smoothing

        logging.info("name " +  constant.args.name)

        training_pass = 0
        
        train_sampler.shuffle(start_epoch)
        
        epoch = start_epoch - 1
        CER = 100000000
        threshhold = 1
        scaler = GradScaler()		# Tensor Cores

        while CER > threshhold:
            epoch = epoch + 1
            sys.stdout.flush()
            total_loss, total_cer, total_wer, total_char, total_word = 0, 0, 0, 0, 0

            start_iter = 0
            if just_once:
                training_results = []
            logging.info("TRAIN")
            model.train()
            pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
            for i, (data) in enumerate(pbar, start=start_iter):
                src, tgt, src_percentages, src_lengths, tgt_lengths = data

                if constant.USE_CUDA:
                    src = src.cuda()
                    tgt = tgt.cuda()

                opt.zero_grad()

                pred, gold, hyp_seq, gold_seq = model(src, src_lengths, tgt, verbose=False)

                try: # handle case for CTC
                    strs_gold, strs_hyps = [], []
                    for ut_gold in gold_seq:
                        str_gold = ""
                        for x in ut_gold:
                            if int(x) == constant.PAD_TOKEN:
                                break
                            str_gold = str_gold + id2label[int(x)]
                        strs_gold.append(str_gold)
                    for ut_hyp in hyp_seq:
                        str_hyp = ""
                        for x in ut_hyp:
                            if int(x) == constant.PAD_TOKEN:
                                break
                            str_hyp = str_hyp + id2label[int(x)]
                        str_hyp = ' '.join([x.strip() for x in str_hyp.split(' ')])
                        strs_hyps.append(str_hyp)
                except Exception as e:
                    print(e)
                    logging.info("NaN predictions")
                    continue

                seq_length = pred.size(1)
                sizes = Variable(src_percentages.mul_(int(seq_length)).int(), requires_grad=False)

                loss, num_correct = calculate_metrics(
                    pred, gold, input_lengths=sizes, target_lengths=tgt_lengths, smoothing=smoothing, loss_type=loss_type)

                if loss.item() == float('Inf'):
                    logging.info("Found infinity loss, masking")
                    loss = torch.where(loss != loss, torch.zeros_like(loss), loss) # NaN masking
                    continue

                if constant.args.verbose:
                     logging.info("GOLD", strs_gold)
                     logging.info("HYP", strs_hyps)

                for j in range(len(strs_hyps)):
                    strs_hyps[j] = strs_hyps[j].replace(constant.SOS_CHAR, '').replace(constant.EOS_CHAR, '')
                    strs_gold[j] = strs_gold[j].replace(constant.SOS_CHAR, '').replace(constant.EOS_CHAR, '')
                    cer = calculate_cer(strs_hyps[j].replace(' ', ''), strs_gold[j].replace(' ', ''))
                    wer = calculate_wer(strs_hyps[j], strs_gold[j])
                    # print(f"GOLD: |{strs_gold[j]}|; HYP: |{strs_hyps[j]}|")
                    if just_once:
                        training_results.append((strs_hyps[j], strs_gold[j], cer, wer))
                    total_cer += cer
                    total_wer += wer
                    total_char += len(strs_gold[j].replace(' ', ''))
                    total_word += len(strs_gold[j].split(" "))

                # loss.backward()
                scaler.scale(loss).backward()	# Tensor Cores

                if constant.args.clip:
                    scaler.unscale_(opt)	# Tensor Cores
                    torch.nn.utils.clip_grad_norm_(model.parameters(), constant.args.max_norm)
                
                # opt.step()
                scaler.step(opt)		# Tensor Cores
                scaler.update()			# Tensor Cores

                total_loss += loss.item()
                non_pad_mask = gold.ne(constant.PAD_TOKEN)
                num_word = non_pad_mask.sum().item()

                TRAIN_LOSS=total_loss/(i+1)
                CER = total_cer*100/max(1,total_char)

                pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} CER:{:.2f}% LR:{:.7f}".format(
                    (epoch+1), TRAIN_LOSS, CER, opt._rate))
                self.writer.add_scalar("Loss/train", TRAIN_LOSS, training_pass+1)
                self.writer.add_scalar("CER/train", CER, training_pass+1)
                self.writer.flush()
                training_pass += 1

            CER = total_cer*100/max(1,total_char)
            logging.info("(Epoch {}) TRAIN LOSS:{:.4f} CER:{:.2f}% LR:{:.7f}".format(
                (epoch+1), total_loss/(len(train_loader)), CER, opt._rate))

            metrics = {}
            metrics["train_loss"] = total_loss / len(train_loader)
            metrics["train_cer"] = total_cer
            metrics["train_wer"] = total_wer
            metrics["history"] = history
            history.append(metrics)

            if epoch % constant.args.save_every == 0:
                save_model(model, (epoch+1), opt, metrics,
                        label2id, id2label, best_model=False)

                save_model(model, (epoch+1), opt, metrics,
                           label2id, id2label, best_model=True)
            
            train_sampler.shuffle(epoch)

            if just_once:
                return training_results

        if epoch % constant.args.save_every != 0:
            save_model(model, (epoch+1), opt, metrics, label2id, id2label, best_model=False)
            save_model(model, (epoch+1), opt, metrics, label2id, id2label, best_model=True)

    def infer (self, model, train_loader, train_sampler,  label2id, id2label):
        """
        Training
        args:
            model: Model object
            train_loader: DataLoader object of the training set
        """
        print("WELCOME TO TRANS WORLD AIRWAYS")
        infer_results = []
        logging.info("EVAL")
        model.eval()
        pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
        for i, (data) in enumerate(pbar):
            src, tgt, src_percentages, src_lengths, tgt_lengths = data

            if constant.USE_CUDA:
                src = src.cuda()
                tgt = tgt.cuda()

            with torch.no_grad():
                pred, gold, hyp_seq, gold_seq = model(src, src_lengths, tgt, verbose=False)

            try: # handle case for CTC
                strs_gold, strs_hyps = [], []
                for ut_gold in gold_seq:
                    str_gold = ""
                    for x in ut_gold:
                        if int(x) == constant.PAD_TOKEN:
                            break
                        str_gold = str_gold + id2label[int(x)]
                    strs_gold.append(str_gold)
                for ut_hyp in hyp_seq:
                    str_hyp = ""
                    for x in ut_hyp:
                        if int(x) == constant.PAD_TOKEN:
                            break
                        str_hyp = str_hyp + id2label[int(x)]
                    str_hyp = ' '.join([x.strip() for x in str_hyp.split(' ')])
                    strs_hyps.append(str_hyp)
            except Exception as e:
                print(e)
                logging.info("NaN predictions")
                continue

            for j in range(len(strs_hyps)):
                strs_hyps[j] = strs_hyps[j].replace(constant.SOS_CHAR, '').replace(constant.EOS_CHAR, '')
                strs_gold[j] = strs_gold[j].replace(constant.SOS_CHAR, '').replace(constant.EOS_CHAR, '')
                cer = calculate_cer(strs_hyps[j].replace(' ', ''), strs_gold[j].replace(' ', ''))
                wer = calculate_wer(strs_hyps[j], strs_gold[j])
                infer_results.append((strs_hyps[j], strs_gold[j], cer, wer))

        return infer_results
