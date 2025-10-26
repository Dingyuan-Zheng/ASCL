from __future__ import print_function, absolute_import
import time

from .evaluation_metrics import accuracy
from .loss import AALS, PGLR, SoftTripletLoss, CrossEntropyLabelSmooth, AdaSPLoss, MaximumMeanDiscrepancy
from .utils.meters import AverageMeter
import torch


class PPLRTrainer(object):
    def __init__(self, model, score, num_class=500, num_part=6, beta=0.5, aals_epoch=5):
        super(PPLRTrainer, self).__init__()
        self.model = model
        self.score = score

        self.num_class = num_class
        self.num_part = num_part
        self.aals_epoch = aals_epoch

        self.criterion_pglr = PGLR(lam=beta).cuda()
        self.criterion_aals = AALS().cuda()
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes=num_class).cuda()
        self.criterion_tri = SoftTripletLoss().cuda()
        # zdy
        self.criterion_adasp = AdaSPLoss().cuda()

    def train(self, epoch, train_dataloader, optimizer, print_freq=1, train_iters=200):
        self.model.train()

        batch_time = AverageMeter()
        losses_gce = AverageMeter()
        losses_tri = AverageMeter()
        losses_pce = AverageMeter()
        precisions = AverageMeter()
        # zdy
        losses_adasp = AverageMeter()

        time.sleep(1)
        end = time.time()
        for i in range(train_iters):
            data = train_dataloader.next()
            inputs, targets, ca = self._parse_data(data)

            # feedforward
            emb_g, emb_p, logits_g, logits_p = self.model(inputs)
            logits_g, logits_p = logits_g[:, :self.num_class], logits_p[:, :self.num_class, :]

            # loss
            loss_gce = self.criterion_pglr(logits_g, logits_p, targets, ca)
            loss_tri = self.criterion_tri(emb_g, targets)

            loss_pce = 0.
            if self.num_part > 0:
                if epoch >= self.aals_epoch:
                    for part in range(self.num_part):
                        loss_pce += self.criterion_aals(logits_p[:, :, part], targets, ca[:, part])
                else:
                    for part in range(self.num_part):
                        loss_pce += self.criterion_ce(logits_p[:, :, part], targets)
                loss_pce /= self.num_part

             # zdy
            loss_adasp = self.criterion_adasp(emb_g, targets)

            loss = (loss_gce + loss_pce) * 1.0 + 1.0 * loss_adasp # ours
            # loss = loss_gce + loss_pce # base

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # summing-up
            prec, = accuracy(logits_g.data, targets.data)

            losses_gce.update(loss_gce.item())
            losses_tri.update(loss_tri.item())
            losses_pce.update(loss_pce.item())
            precisions.update(prec[0])
            # zdy
            losses_adasp.update(loss_adasp.item())

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'L_GCE {:.3f} ({:.3f})\t'
                      'L_PCE {:.3f} ({:.3f})\t'
                      'L_TRI {:.3f} ({:.3f})\t'
                      'L_AdaSP {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(train_dataloader),
                              batch_time.val, batch_time.avg,
                              losses_gce.val, losses_gce.avg,
                              losses_pce.val, losses_pce.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_adasp.val, losses_adasp.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, idxs = inputs
        ca = self.score[idxs]
        return imgs.cuda(), pids.cuda(), ca.cuda()


class PPLRTrainerCAM(object):
    def __init__(self, model, score, memory, memory_p, num_class=500, num_part=6, beta=0.5, aals_epoch=5, lam_cam=0.5):
        super(PPLRTrainerCAM, self).__init__()
        self.model = model
        self.score = score
        self.memory = memory
        self.memory_p = memory_p

        self.num_class = num_class
        self.num_part = num_part
        self.lam_cam = lam_cam
        self.aals_epoch = aals_epoch

        self.criterion_pglr = PGLR(lam=beta).cuda()
        self.criterion_aals = AALS().cuda()
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes=num_class).cuda()
        self.criterion_tri = SoftTripletLoss().cuda()
        # zdy
        self.criterion_adasp = AdaSPLoss().cuda()
        self.mmd = MaximumMeanDiscrepancy().cuda()

    def train(self, epoch, train_dataloader, optimizer, print_freq=1, train_iters=200):
        self.model.train()

        batch_time = AverageMeter()
        losses_gce = AverageMeter()
        losses_tri = AverageMeter()
        losses_cam = AverageMeter()
        losses_pce = AverageMeter()
        # zdy
        losses_adasp = AverageMeter()

        precisions = AverageMeter()

        time.sleep(1)
        end = time.time()
        for i in range(train_iters):
            data = train_dataloader.next()
            inputs, targets, cams, ca = self._parse_data(data)

            # feedforward
            emb_g, emb_p, logits_g, logits_p = self.model(inputs)
            logits_g, logits_p = logits_g[:, :self.num_class], logits_p[:, :self.num_class, :]

            # loss
            loss_gce = self.criterion_pglr(logits_g, logits_p, targets, ca)
            loss_tri = self.criterion_tri(emb_g, targets)
            loss_gcam = self.memory(emb_g, targets, cams)

            loss_pce = 0.
            loss_pcam = 0.
            if self.num_part > 0:
                if epoch >= self.aals_epoch:
                    for part in range(self.num_part):
                        loss_pce += self.criterion_aals(logits_p[:, :, part], targets, ca[:, part])
                        loss_pcam += self.memory_p[part](emb_p[:, :, part], targets, cams)
                else:
                    for part in range(self.num_part):
                        loss_pce += self.criterion_ce(logits_p[:, :, part], targets)
                        loss_pcam += self.memory_p[part](emb_p[:, :, part], targets, cams)
                loss_pce /= self.num_part
                loss_pcam /= self.num_part

            loss_cam = loss_pcam + loss_gcam
            # zdy
            loss_adasp = self.criterion_adasp(emb_g, targets)

            loss = loss_gce + loss_pce + loss_cam * self.lam_cam + 1.0 * loss_adasp


            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # summing-up
            prec, = accuracy(logits_g.data, targets.data)

            losses_gce.update(loss_gce.item())
            losses_tri.update(loss_tri.item())
            losses_cam.update(loss_cam.item())
            losses_pce.update(loss_pce.item())
            # zdy
            losses_adasp.update(loss_adasp.item())

            precisions.update(prec[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'L_GCE {:.3f} ({:.3f})\t'
                      'L_PCE {:.3f} ({:.3f})\t'
                      'L_TRI {:.3f} ({:.3f})\t'
                      'L_CAM {:.3f} ({:.3f})\t'
                      'L_AdaSP {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(train_dataloader),
                              batch_time.val, batch_time.avg,
                              losses_gce.val, losses_gce.avg,
                              losses_pce.val, losses_pce.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_cam.val, losses_cam.avg,
                              losses_adasp.val, losses_adasp.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, cids, idxs = inputs
        ca = self.score[idxs]
        return imgs.cuda(), pids.cuda(), cids.cuda(), ca.cuda()


