"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
import models.networks as networks
import util.util as util
from models.networks.loss import losses_computer
import warnings

warnings.filterwarnings("ignore")


class Pix2PixModel(nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = jt.float32
        self.ByteTensor = jt.float32

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            if opt.netD == "dpgan":
                self.losses_computer = losses_computer(self.opt)
                self.criterionGAN = networks.GANLoss(
                    opt.gan_mode, tensor=self.FloatTensor, opt=self.opt
                )
                if not opt.no_vgg_loss:
                    self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
                self.criterionFeat = nn.MSELoss()
                self.MSELoss = nn.MSELoss(reduction="mean")
                # Todo: 在这里完善 dpgan中的几个loss, dpgan的GANloss 可能与gaugan相同，能直接用
            else:
                self.criterionGAN = networks.GANLoss(
                    opt.gan_mode, tensor=self.FloatTensor, opt=self.opt
                )
                self.criterionFeat = nn.L1Loss()
                if not opt.no_vgg_loss:
                    self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
                if opt.use_vae:
                    self.KLDLoss = networks.KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def execute(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data)
        # print("input_semantics: " , input_semantics.shape)
        # print("real_image: ", real_image.shape)
        # exit(0)

        if mode == "generator":
            g_loss, generated = self.compute_generator_loss(input_semantics, real_image)
            return g_loss, generated
        elif mode == "discriminator":
            d_loss = self.compute_discriminator_loss(input_semantics, real_image)
            return d_loss
        elif mode == "encode_only":
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == "inference":
            with jt.no_grad():
                fake_image, _ = self.generate_fake(input_semantics, real_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = jt.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = jt.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, "G", epoch, self.opt)
        util.save_network(self.netD, "D", epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, "E", epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, "G", opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, "D", opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, "E", opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # change data types
        data["label"] = data["label"].long()

        # create one-hot label map
        label_map = data["label"]
        bs, _, h, w = label_map.size()
        nc = (
            self.opt.label_nc + 1
            if self.opt.contain_dontcare_label
            else self.opt.label_nc
        )
        input_label = jt.zeros((bs, nc, h, w), dtype=self.FloatTensor)
        input_semantics = input_label.scatter_(1, label_map, jt.float32(1.0))

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data["instance"]
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = jt.concat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data["image"]

    def align_loss(self, feats, feats_ref):
        loss_align = 0
        for f, fr in zip(feats, feats_ref):
            loss_align += self.criterionFeat(f, fr)
        loss_align = loss_align * self.opt.lambda_feat
        return loss_align

    def compute_generator_loss(self, input_semantics, real_image):
        """'
        将dpgan和gangan的g_loss分开计算
        """
        if self.opt.netD == "dpgan":
            loss_G, fake_image = self.compute_G_loss_dpgan(
                input_semantics=input_semantics, real_image=real_image
            )
            # Todo: 完善dpgan G_loss
            return loss_G, fake_image
        else:
            G_losses, fake_image = self.compute_G_loss_gaugan(
                self, input_semantics, real_image
            )
            return G_losses, fake_image

    def compute_G_loss_dpgan(self, input_semantics, real_image):
        G_losses = {}

        fake = self.netG(input_semantics)
        # if self.opt.use_vae:
        #     G_losses["KLD"] = KLD_loss

        output_D, scores, feats = self.netD(fake)
        _, _, feats_ref = self.netD(real_image)
        loss_G_adv = self.losses_computer.loss(
            input=output_D, label=input_semantics, for_real=True
        )
        G_losses["adv"] = loss_G_adv
        loss_ms = self.criterionGAN(scores, True, for_discriminator=False)
        G_losses["ms"] = loss_ms
        loss_align = self.align_loss(feats, feats_ref)
        G_losses["align"] = loss_align
        if not self.opt.no_vgg_loss:
            loss_G_vgg = self.opt.lambda_vgg * self.criterionVGG(fake, real_image)
            G_losses["vgg"] = loss_G_vgg
        else:
            loss_G_vgg = None
        return G_losses, fake

    def compute_G_loss_gaugan(self, input_semantics, real_image):
        G_losses = {}

        fake_image, KLD_loss = self.generate_fake(
            input_semantics, real_image, compute_kld_loss=self.opt.use_vae
        )

        if self.opt.use_vae:
            G_losses["KLD"] = KLD_loss

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image
        )

        G_losses["GAN"] = self.criterionGAN(pred_fake, True, for_discriminator=False)

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(0.0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach()
                    )
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses["GAN_Feat"] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses["VGG"] = (
                self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg
            )

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image):
        if self.opt.netD == "dpgan":
            D_losses = self.compute_D_loss_dpgan(input_semantics, real_image)
            return D_losses
        else:
            D_losses = self.compute_D_loss_gaugan(input_semantics, real_image)
            return D_losses

    def compute_D_loss_dpgan(self, input_semantics, real_image):
        """

        Args:
            input_semantics: one-hot
            real_image:

        Returns:
            loss_D: sum of all losses
            D_losses: dict of all losses

        """
        D_losses = {}
        with jt.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image)
            # fake_image = fake_image.detach()
            # fake_image.requires_grad_()

        output_D_fake, scores_fake, _ = self.netD(fake_image)
        loss_D_fake = self.losses_computer.loss(
            input=output_D_fake, label=input_semantics, for_real=False
        )
        loss_ms_fake = self.criterionGAN(scores_fake, False, for_discriminator=True)
        D_losses["adv_fake"] = loss_D_fake
        D_losses["ms_fake"] = loss_ms_fake
        output_D_real, scores_real, _ = self.netD(real_image)
        loss_D_real = self.losses_computer.loss(
            input=output_D_real, label=input_semantics, for_real=True
        )
        loss_ms_real = self.criterionGAN(scores_real, True, for_discriminator=True)
        D_losses["adv_real"] = loss_D_real
        D_losses["ms_real"] = loss_ms_real
        if not self.opt.no_labelmix:
            mixed_inp, mask = generate_labelmix(input_semantics, fake_image, real_image)
            output_D_mixed, _, _ = self.netD(mixed_inp)
            loss_D_lm = self.opt.lambda_labelmix * self.losses_computer.loss_labelmix(
                mask=mask,
                output_D_mixed=output_D_mixed,
                output_D_fake=output_D_fake,
                output_D_real=output_D_real,
            )
            D_losses["lm_loss"] = loss_D_lm

        return D_losses

    def compute_D_loss_gaugan(self, input_semantics, real_image):
        D_losses = {}
        with jt.no_grad():
            fake_image, _ = self.generate_fake(input_semantics, real_image)
            # fake_image = fake_image.detach()
            # fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image
        )

        D_losses["D_Fake"] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses["D_real"] = self.criterionGAN(pred_real, True, for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        # if self.opt.use_vae and self.opt.isTrain:
        z, mu, logvar = self.encode_z(real_image)
        if compute_kld_loss:
            KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image = self.netG(input_semantics, z=z)

        assert (
            not compute_kld_loss
        ) or self.opt.use_vae, "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = jt.concat([input_semantics, fake_image], dim=1)
        real_concat = jt.concat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = jt.concat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[: tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2 :] for tensor in p])
        else:
            fake = pred[: pred.size(0) // 2]
            real = pred[pred.size(0) // 2 :]

        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = jt.exp(0.5 * logvar)
        eps = jt.randn_like(std)
        return eps.multiply(std).add(mu)

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0


def generate_labelmix(label, fake_image, real_image):
    target_map = jt.argmax(label, dim=1, keepdims=True)[0]
    all_classes = jt.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = jt.randint(0, 2, (1,))
    target_map = target_map.float()
    mixed_image = target_map * real_image + (1 - target_map) * fake_image
    return mixed_image, target_map
