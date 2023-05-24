from dataset import *
from model import *
from utils import *
from learn import *
import argparse
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
import math
from pandas import DataFrame
from utils import read_config

def run(data, args, data2):
    pbar = tqdm(range(args.runs), unit='run')
    criterion = nn.BCELoss()
    if args.ood == 4:
        acc, f1, auc_roc, parity, equality = np.zeros([args.runs,len(data2)]), np.zeros([args.runs,len(data2)]), np.zeros([args.runs,len(data2)]), np.zeros([args.runs,len(data2)]), np.zeros([args.runs, len(data2)])
    elif args.ood == 3:
        acc, f1, auc_roc, parity, equality = np.zeros([args.runs,len(args.strlist)]), np.zeros([args.runs,len(args.strlist)]), np.zeros([args.runs,len(args.strlist)]), np.zeros([args.runs,len(args.strlist)]), np.zeros([args.runs, len(args.strlist)])

    else:
        acc, f1, auc_roc, parity, equality = np.zeros(args.runs), np.zeros(
        args.runs), np.zeros(args.runs), np.zeros(args.runs), np.zeros(args.runs)

    data = data.to(args.device)

    if args.ood == 4:
        for i in range(len(data2)):
            # if i == args.inidIndex:
            #     data2[i] = data2[i].to(args.device)
            #     continue
            data2[i] = data2[i].to(args.device)
            data2[i].test_mask = data2[i].test_mask | data2[i].val_mask | data2[i].test_mask
    elif args.ood == 2:
        data2 = data2.to(args.device)
    elif data2 != None:
        data2 = data2.to(args.device)
    else:
        data2 = data2

    generator = channel_masker(args).to(args.device)
    graphEdit = Graph_Editer(1, data.x.shape[1], args.device).to(args.device)

    optimizer_g = torch.optim.Adam([
        dict(params=generator.weights, weight_decay=args.g_wd)], lr=args.g_lr)
    optimizer_gF = torch.optim.Adam([
        dict(params=graphEdit.transFeature.parameters(), weight_decay=args.g_wd), ], lr=args.g_lr)

    optimizer_align = torch.optim.Adam([
        dict(params=generator.weights, weight_decay=args.g_wd)], lr=args.g_lr)

    discriminator = MLP_discriminator(args).to(args.device)
    optimizer_d = torch.optim.Adam([
        dict(params=discriminator.lin.parameters(), weight_decay=args.d_wd)], lr=args.d_lr)

    classifier = MLP_classifier(args).to(args.device)
    optimizer_c = torch.optim.Adam([
        dict(params=classifier.lin.parameters(), weight_decay=args.c_wd)], lr=args.c_lr)

    if (args.encoder == 'MLP'):
        encoder = MLP_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.lin.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
    elif (args.encoder == 'GCN'):
        if args.prop == 'scatter':
            encoder = GCN_encoder_scatter(args).to(args.device)
        else:
            encoder = GCN_encoder_spmm(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.lin.parameters(), weight_decay=args.e_wd),
            dict(params=encoder.bias, weight_decay=args.e_wd)], lr=args.e_lr)
    elif (args.encoder == 'GIN'):
        encoder = GIN_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.conv.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)
    elif (args.encoder == 'SAGE'):
        encoder = SAGE_encoder(args).to(args.device)
        optimizer_e = torch.optim.Adam([
            dict(params=encoder.conv1.parameters(), weight_decay=args.e_wd),
            dict(params=encoder.conv2.parameters(), weight_decay=args.e_wd)], lr=args.e_lr)

    t_idx_s0 = data.sens[data.train_mask] == 0
    t_idx_s1 = data.sens[data.train_mask] == 1
    t_idx_s0_y1 = torch.logical_and(t_idx_s0, data.y[data.train_mask] == 1)
    t_idx_s1_y1 = torch.logical_and(t_idx_s1, data.y[data.train_mask] == 1)
    t_idx_s0_y0 = torch.logical_and(t_idx_s0, data.y[data.train_mask] == 0)
    t_idx_s1_y0 = torch.logical_and(t_idx_s1, data.y[data.train_mask] == 0)
    t_num_s0_y1, t_num_s1_y1, t_num_s0_y0, t_num_s1_y0, = sum(t_idx_s0_y1), sum(t_idx_s1_y1), sum(t_idx_s0_y0), sum(t_idx_s1_y0),

    idx_s0 = data.sens == 0
    idx_s1 = data.sens == 1
    idx_s0_y1 = torch.logical_and(idx_s0, data.y == 1)
    idx_s1_y1 = torch.logical_and(idx_s1, data.y == 1)
    idx_s0_y0 = torch.logical_and(idx_s0, data.y == 0)
    idx_s1_y0 = torch.logical_and(idx_s1, data.y == 0)
    num_s0_y1, num_s1_y1, num_s0_y0, num_s1_y0, = sum(idx_s0_y1), sum(idx_s1_y1), sum(idx_s0_y0), sum(idx_s1_y0),


    eweight = torch.ones(data.edge_index.shape[1]).to(data.x.device)
    adj = torch.sparse_coo_tensor(data.edge_index, eweight, [data.x.shape[0], data.x.shape[0]])
    A2 = torch.spmm(adj, adj)

    for count in pbar:
        seed_everything(count + args.seed)
        graphEdit.reset_parameters()
        discriminator.reset_parameters()
        classifier.reset_parameters()
        encoder.reset_parameters()

        best_val_tradeoff = 0
        best_val_loss = math.inf

        for epoch in range(0, args.epochs):
            print(f"======={epoch}=======")

            # train discriminator to recognize the sensitive group / make group close
            if args.discri == 1:
                discriminator.train()
                for epoch_d in range(0, args.dic_epochs):
                    optimizer_d.zero_grad()

                    h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                    output = discriminator(h)

                    loss_d = criterion(output.view(-1),
                                       data.x[:, args.sens_idx])

                    loss_d.backward()
                    optimizer_d.step()
                discriminator.eval()

            # train classifier and encoder
            classifier.train()
            encoder.train()
            for epoch_c in range(0, args.cla_epochs):
                # print("classify")
                optimizer_c.zero_grad()
                optimizer_e.zero_grad()

                h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                output = classifier(h)


                if args.discri == 1:
                    loss_c = F.binary_cross_entropy_with_logits(output[data.train_mask], data.y[data.train_mask].unsqueeze(1)).to(args.device) # - (1-args.labda) * criterion(output2.view(-1), data.x[:, args.sens_idx])
                else:
                    loss_c = F.binary_cross_entropy_with_logits(output[data.train_mask], data.y[data.train_mask].unsqueeze(1)).to(args.device)

                loss_c.backward()

                optimizer_e.step()
                optimizer_c.step()
            encoder.eval()

            # train generator to fool discriminator
            generator.train()
            encoder.train()
            discriminator.eval()
            for epoch_g in range(0, args.g_epochs):
                optimizer_g.zero_grad()
                optimizer_e.zero_grad()

                if (args.f_mask == 'yes'):
                    loss_g = 0
                    feature_weights = generator()
                    for k in range(args.K):
                        mask = F.gumbel_softmax(
                            feature_weights, tau=1, hard=False)[:, 0]

                        x = data.x * mask
                        h = encoder(x, data.edge_index, data.adj_norm_sp)
                        output = discriminator(h)

                        loss_g += F.mse_loss(output.view(-1),
                                             0.5 * torch.ones_like(output.view(-1))) + args.ratio * F.mse_loss(
                            mask.view(-1), torch.ones_like(mask.view(-1)))

                    loss_g = loss_g / args.K
                else:
                    h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                    output = discriminator(h)

                    loss_g = F.mse_loss(output.view(-1),
                                        0.5 * torch.ones_like(output.view(-1)))

                loss_g.backward()

                optimizer_g.step()
                optimizer_e.step()




            if args.close == 1:
                for epoch_d in range(0, args.clo_epochs):
                    encoder.train()
                    optimizer_e.zero_grad()

                    h = encoder(data.x, data.edge_index, data.adj_norm_sp)
                    h0 = F.normalize(h[data.sens == 0])
                    h1 = F.normalize(h[data.sens == 1])
                    loss_e = -torch.mean(torch.mm(h0, h1.T))

                    loss_e.backward()
                    optimizer_e.step()

            # args.disturb = 1
            # args.modiStru = 1
            if args.disturb == 1 and epoch > args.start:
                graphEdit.train()
                encoder.eval()
                classifier.eval()
                if epoch % 10 == 0 and args.modiStru == 1:
                    if epoch % 20 == 0:
                        edge_index2 = graphEdit.modify_structure1(data.edge_index, adj, A2, data.sens, data.x.shape[0], args.drope_rate)
                    else:
                        edge_index2 = graphEdit.modify_structure2(data.edge_index, adj, A2, data.sens, data.x.shape[0], args.drope_rate)
                else:
                    edge_index2 = data.edge_index
                for epoch_g in range(0, args.dtb_epochs):
                    optimizer_gF.zero_grad()
                    x2 = graphEdit(data.x)
                    h2 = encoder(x2, edge_index2, data.adj_norm_sp)
                    h2 = F.normalize(h2)
                    output2 = classifier(h2)
                    loss_edit2 = -fair_metric2(output2[data.train_mask], data.y[data.train_mask], t_idx_s0_y1, t_idx_s1_y1,
                                              t_num_s0_y1, t_num_s1_y1)
                    loss_edit2.backward()

                    optimizer_gF.step()

            # shift align
            if args.disturb == 1 and args.align == 1 and epoch > args.start:
                graphEdit.eval()
                encoder.train()
                classifier.train()
                x2 = graphEdit(data.x).detach()
                for epoch_a in range(0, args.a_epochs):
                    optimizer_e.zero_grad()
                    optimizer_c.zero_grad()
                    h2 = encoder(x2, edge_index2, data.adj_norm_sp)
                    h1 = encoder(data.x, data.edge_index, data.adj_norm_sp)
                    h2 = F.normalize(h2)
                    h1 = F.normalize(h1)


                    loss_align = - (data.x.shape[0]) / (num_s0_y0)  * torch.mean(torch.mm(h1[idx_s0_y0], h2[idx_s0_y0].T)) \
                                 - (data.x.shape[0]) / (num_s0_y1) * torch.mean(torch.mm(h1[idx_s0_y1], h2[idx_s0_y1].T)) \
                                 - (data.x.shape[0]) / (num_s1_y0) * torch.mean(torch.mm(h1[idx_s1_y0], h2[idx_s1_y0].T)) \
                                 - (data.x.shape[0]) / (num_s1_y1) * torch.mean(torch.mm(h1[idx_s1_y1], h2[idx_s1_y1].T))

                    loss_align = loss_align * 0.01
                    loss_align.backward(retain_graph=True)

                    optimizer_e.step()
                    optimizer_c.step()

            "=====test======="
            if args.ood == 3:
                test_acc = [0 for n in range(len(args.strlist))]
                best_val_tradeoff = [0 for n in range(len(args.strlist))]
                test_auc_roc = [0 for n in range(len(args.strlist))]
                test_f1 = [0 for n in range(len(args.strlist))]
                test_parity = [0 for n in range(len(args.strlist))]
                test_equality = [0 for n in range(len(args.strlist))]
            elif args.ood == 4:
                test_acc = [0 for n in range(len(data2))]
                best_val_tradeoff = [0 for n in range(len(data2))]
                test_auc_roc = [0 for n in range(len(data2))]
                test_f1 = [0 for n in range(len(data2))]
                test_parity = [0 for n in range(len(data2))]
                test_equality = [0 for n in range(len(data2))]



            if args.ood == 1 or args.ood == 2:
                accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged3(
                    data2.x, classifier, discriminator, generator, encoder, data2, args)
                if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (
                        tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
                    test_acc = accs['test']
                    test_auc_roc = auc_rocs['test']
                    test_f1 = F1s['test']
                    test_parity, test_equality = tmp_parity['test'], tmp_equality['test']

                    best_val_tradeoff = auc_rocs['val'] + F1s['val'] + \
                                        accs['val'] - (tmp_parity['val'] + tmp_equality['val'])

            elif args.ood == 4:
                for i in range(len(data2)):
                    accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged3(
                        data2[i].x, classifier, discriminator, generator, encoder, data2[i], args)


                    if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (
                            tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff[i]:
                        test_acc[i] = accs['test']
                        test_auc_roc[i] = auc_rocs['test']
                        test_f1[i] = F1s['test']
                        test_parity[i], test_equality[i] = tmp_parity['test'], tmp_equality['test']

                        best_val_tradeoff[i] = auc_rocs['val'] + F1s['val'] + \
                                            accs['val'] - (tmp_parity['val'] + tmp_equality['val'])
            elif args.ood == 3:
                if epoch != (args.epochs - 1):
                    continue
                for i in range(len(args.strlist)):
                    datatmp, _, _, _, _, _ = get_dataset(args.dataset, args.outid + args.strlist[i], args.top_k)
                    datatmp = datatmp.to(args.device)
                    datatmp.test_mask = datatmp.test_mask | datatmp.val_mask | datatmp.test_mask
                    accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged3(
                        datatmp.x, classifier, discriminator, generator, encoder, datatmp, args)



                    test_acc[i] = accs['test']
                    test_auc_roc[i] = auc_rocs['test']
                    test_f1[i] = F1s['test']
                    test_parity[i], test_equality[i] = tmp_parity['test'], tmp_equality['test']



            else:
                accs, auc_rocs, F1s, tmp_parity, tmp_equality = evaluate_ged3(
                    data.x, classifier, discriminator, generator, encoder, data, args)
                if auc_rocs['val'] + F1s['val'] + accs['val'] - args.alpha * (
                        tmp_parity['val'] + tmp_equality['val']) > best_val_tradeoff:
                    test_acc = accs['test']
                    test_auc_roc = auc_rocs['test']
                    test_f1 = F1s['test']
                    test_parity, test_equality = tmp_parity['test'], tmp_equality['test']

                    best_val_tradeoff = auc_rocs['val'] + F1s['val'] + \
                                        accs['val'] - (tmp_parity['val'] + tmp_equality['val'])







        if args.ood == 3 or args.ood == 4:
            for i in range(len(args.strlist)):
                acc[count][i] = test_acc[i]
                f1[count][i] = test_f1[i]
                auc_roc[count][i] = test_auc_roc[i]
                parity[count][i] = test_parity[i]
                equality[count][i] = test_equality[i]
        else:
            acc[count] = test_acc
            f1[count] = test_f1
            auc_roc[count] = test_auc_roc
            parity[count] = test_parity
            equality[count] = test_equality


    return acc, f1, auc_roc, parity, equality

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='german')
    parser.add_argument('--inid', type=str, default='')
    parser.add_argument('--outid', type=str, default='')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--start', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--dic_epochs', type=int, default=2)
    parser.add_argument('--dtb_epochs', type=int, default=5)
    # parser.add_argument('--g_epochs', type=int, default=5)
    parser.add_argument('--cla_epochs', type=int, default=10)
    parser.add_argument('--clo_epochs', type=int, default=2)
    parser.add_argument('--a_epochs', type=int, default=5)
    parser.add_argument('--g_epochs', type=int, default=5)


    parser.add_argument('--g_lr', type=float, default=0.001)
    parser.add_argument('--g_wd', type=float, default=0)
    parser.add_argument('--d_lr', type=float, default=0.001)
    parser.add_argument('--d_wd', type=float, default=0)
    parser.add_argument('--c_lr', type=float, default=0.005)
    parser.add_argument('--c_wd', type=float, default=0)
    parser.add_argument('--e_lr', type=float, default=0.005)
    parser.add_argument('--e_wd', type=float, default=0)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--prop', type=str, default='scatter')
    # parser.add_argument('--predictfile', type=str, default='vgnn319')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='GCN')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--clip_e', type=float, default=1)
    parser.add_argument('--clip_c', type=float, default=1)
    parser.add_argument('--f_mask', type=str, default='no')
    parser.add_argument('--weight_clip', type=str, default='yes')
    parser.add_argument('--ratio', type=float, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--ood', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--close', type=int, default=1)
    parser.add_argument('--discri', type=int, default=1)
    parser.add_argument('--dropf', type=int, default=0)
    parser.add_argument('--dropf_rate', type=float, default=0.1)
    parser.add_argument('--disturb', type=int, default=1)
    parser.add_argument('--align', type=int, default=1)
    parser.add_argument('--modiStru', type=int, default=0)
    parser.add_argument('--drope_rate', type=float, default=0.5)
    parser.add_argument('--tune', type=str, default='True', help='if tune')
    parser.add_argument('--times', type=str, default='bail')
    # parser.add_argument('--configfile', type=str, default='1111', help='configfile')
    parser.add_argument('--labda', type=float, default=0.5)



    args = parser.parse_args()
    args.strlist = None
    if args.tune == 'True':
        args = read_config(args)
    if args.outid == "all":
        args.outid = ""
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(args)
    data, args.sens_idx, args.corr_sens, args.corr_idx, args.x_min, args.x_max = get_dataset(
        args.dataset, args.inid, args.top_k)
    if args.ood == 1 or args.ood ==2:
        if (args.dataset == "pokec"):
            data2, _, _, _, _, _ = get_dataset(
                "pokec", args.outid, args.top_k)
            # data2 = None
        elif (args.dataset == "credit"):
            data2, _, _, _, _, _ = get_dataset(
                args.dataset, args.outid, args.top_k)
        elif (args.dataset == "bail"):
            data2, _, _, _, _, _ = get_dataset(
                args.dataset, args.outid, args.top_k)
        else:
            data2 = None
    elif args.ood == 3:
        data2 = []

        if args.dataset == "bail":
            if args.outid == "_md0":
                args.strlist = ['_0.56_0.35_0.54_0.25_0.06_0.56', '_0.51_0.32_0.49_0.24_0.00_0.56',
                           '_0.58_0.36_0.56_0.26_0.10_0.56',
                           '_0.49_0.30_0.47_0.25_0.00_0.56', '_0.63_0.39_0.61_0.30_0.20_0.56',
                           '_0.45_0.27_0.43_0.29_0.00_0.56',
                           '_0.67_0.42_0.65_0.36_0.29_0.56', '_0.41_0.25_0.40_0.33_0.00_0.56',
                           '_0.72_0.45_0.70_0.43_0.39_0.56',
                           '_0.37_0.23_0.37_0.37_0.00_0.44', '_0.76_0.48_0.74_0.51_0.48_0.56',
                           '_0.34_0.21_0.34_0.41_0.00_0.44',
                           '_0.81_0.51_0.79_0.59_0.58_0.56', '_0.31_0.19_0.32_0.44_0.00_0.44',
                           '_0.86_0.54_0.84_0.68_0.68_0.56',
                           '_0.29_0.17_0.30_0.47_0.00_0.44', '_0.90_0.57_0.89_0.78_0.79_0.56',
                           '_0.27_0.16_0.28_0.50_0.00_0.44',
                           '_0.95_0.60_0.94_0.89_0.89_0.56', '_0.25_0.15_0.27_0.52_0.00_0.44',
                           '_0.64_0.40_0.62_0.32_0.22_0.56',
                           '_0.43_0.26_0.42_0.30_0.00_0.56', '_0.69_0.43_0.67_0.38_0.32_0.56',
                           '_0.40_0.24_0.39_0.34_0.00_0.44',
                           '_0.73_0.46_0.71_0.45_0.42_0.56', '_0.36_0.22_0.36_0.38_0.00_0.44',
                           '_0.78_0.49_0.76_0.53_0.51_0.56',
                           '_0.33_0.20_0.34_0.42_0.00_0.44', '_0.82_0.52_0.80_0.62_0.61_0.56',
                           '_0.30_0.18_0.31_0.45_0.00_0.44',
                           '_0.87_0.55_0.86_0.71_0.71_0.56', '_0.28_0.17_0.30_0.48_0.00_0.44',
                           '_0.92_0.58_0.91_0.81_0.82_0.56',
                           '_0.26_0.16_0.28_0.51_0.00_0.44', '_0.97_0.61_0.96_0.92_0.92_0.56',
                           '_0.24_0.14_0.27_0.53_0.00_0.44',
                           '_0.61_0.38_0.59_0.28_0.15_0.56', '_0.47_0.29_0.45_0.27_0.00_0.56',
                           '_0.65_0.41_0.63_0.33_0.24_0.56',
                           '_0.43_0.26_0.41_0.31_0.00_0.56', '_0.70_0.43_0.68_0.39_0.34_0.56',
                           '_0.39_0.24_0.38_0.35_0.00_0.44',
                           '_0.74_0.46_0.72_0.47_0.43_0.56', '_0.36_0.22_0.35_0.39_0.00_0.44',
                           '_0.79_0.49_0.77_0.55_0.53_0.56',
                           '_0.33_0.20_0.33_0.43_0.00_0.44', '_0.83_0.52_0.81_0.64_0.63_0.56',
                           '_0.30_0.18_0.31_0.46_0.00_0.44',
                           '_0.88_0.56_0.86_0.73_0.73_0.56', '_0.28_0.17_0.29_0.48_0.00_0.44',
                           '_0.93_0.59_0.92_0.84_0.84_0.56',
                           '_0.26_0.15_0.28_0.51_0.00_0.44', '_0.98_0.62_0.97_0.94_0.94_0.56',
                           '_0.24_0.14_0.26_0.53_0.00_0.44']
            elif args.outid == "_md3":
                args.strlist = ['_0.60_0.30_0.60_0.25_0.18_0.48', '_0.46_0.23_0.46_0.23_0.00_0.48', '_0.65_0.32_0.64_0.32_0.27_0.48',
                 '_0.42_0.21_0.42_0.28_0.00_0.48', '_0.69_0.35_0.69_0.40_0.37_0.48', '_0.38_0.19_0.39_0.32_0.00_0.48',
                 '_0.74_0.37_0.74_0.48_0.46_0.48', '_0.35_0.17_0.36_0.36_0.00_0.48', '_0.79_0.39_0.78_0.57_0.56_0.48',
                 '_0.32_0.16_0.34_0.40_0.00_0.48', '_0.83_0.42_0.83_0.66_0.66_0.48', '_0.30_0.15_0.32_0.44_0.00_0.48',
                 '_0.88_0.44_0.88_0.76_0.75_0.48', '_0.27_0.13_0.30_0.47_0.00_0.48', '_0.93_0.47_0.93_0.85_0.85_0.48',
                 '_0.25_0.12_0.28_0.49_0.00_0.48', '_0.98_0.49_0.98_0.95_0.95_0.48', '_0.24_0.12_0.27_0.52_0.00_0.48',
                 '_0.56_0.28_0.55_0.21_0.09_0.48', '_0.51_0.25_0.50_0.20_0.02_0.48', '_0.58_0.29_0.57_0.23_0.13_0.48',
                 '_0.49_0.24_0.48_0.21_0.00_0.48', '_0.62_0.31_0.62_0.28_0.23_0.48', '_0.44_0.22_0.44_0.26_0.00_0.48',
                 '_0.67_0.34_0.67_0.36_0.32_0.48', '_0.40_0.20_0.40_0.30_0.00_0.48', '_0.72_0.36_0.72_0.44_0.41_0.48',
                 '_0.37_0.18_0.37_0.34_0.00_0.48', '_0.76_0.38_0.76_0.53_0.51_0.48', '_0.34_0.17_0.35_0.38_0.00_0.48',
                 '_0.81_0.41_0.81_0.62_0.61_0.48', '_0.31_0.15_0.33_0.42_0.00_0.48', '_0.86_0.43_0.85_0.71_0.70_0.48',
                 '_0.29_0.14_0.31_0.45_0.00_0.48', '_0.90_0.45_0.90_0.80_0.80_0.48', '_0.26_0.13_0.29_0.48_0.00_0.48',
                 '_0.95_0.48_0.95_0.90_0.90_0.48', '_0.24_0.12_0.27_0.51_0.00_0.48', '_0.64_0.32_0.64_0.31_0.25_0.48',
                 '_0.43_0.21_0.43_0.27_0.00_0.48', '_0.68_0.34_0.68_0.38_0.35_0.48', '_0.39_0.19_0.39_0.32_0.00_0.48',
                 '_0.73_0.37_0.73_0.46_0.44_0.48', '_0.36_0.18_0.37_0.36_0.00_0.48', '_0.78_0.39_0.78_0.55_0.54_0.48',
                 '_0.33_0.16_0.34_0.40_0.00_0.48', '_0.82_0.41_0.82_0.64_0.64_0.48', '_0.30_0.15_0.32_0.43_0.00_0.48',
                 '_0.87_0.44_0.87_0.74_0.73_0.48', '_0.28_0.14_0.30_0.46_0.00_0.48', '_0.92_0.46_0.92_0.83_0.83_0.48',
                 '_0.26_0.13_0.28_0.49_0.00_0.48', '_0.97_0.49_0.97_0.93_0.93_0.48', '_0.24_0.12_0.27_0.51_0.00_0.48']


        data2 = None
        args.in_hom = [0 for i in range(len(args.strlist))]
        args.edge_hom = [0 for i in range(len(args.strlist))]
        args.node_hom = [0 for i in range(len(args.strlist))]
        args.class_hom = [0 for i in range(len(args.strlist))]
        args.agg_hom = [0 for i in range(len(args.strlist))]

    elif args.ood == 4:
        data2 = []
        if args.dataset == "credit":

            args.strlist = ['_C1', '_C2', '_C3', '_C4']
            # args.inidIndex=args.strlist.index(args.inid)
            for i in range(len(args.strlist)):
                # if args.inidIndex == i:
                #     data2.append(data)
                #     continue
                datatmp, _, _, _, _, _ = get_dataset(
                    args.dataset,  args.strlist[i], args.top_k)
                data2.append(datatmp)
        elif args.dataset == "bail":
            args.strlist = ['_B1',  '_B2', '_B3', '_B4',]
            # args.inidIndex = args.strlist.index(args.inid)
            for i in range(len(args.strlist)):
                # if args.inidIndex == i:
                #     data2.append(data)
                #     continue
                datatmp, _, _, _, _, _ = get_dataset(
                    args.dataset,  args.strlist[i], args.top_k)
                data2.append(datatmp)
        elif args.dataset == "pokec":
            args.strlist = ['_z', '_n',]
            args.inidIndex = args.strlist.index(args.inid)
            for i in range(len(args.strlist)):
                if args.inidIndex == i:
                    data2.append(data)
                    continue
                datatmp, _, _, _, _, _ = get_dataset(
                    args.dataset,  args.strlist[i], args.top_k)
                data2.append(datatmp)


    else:
        data2 = None
    args.num_features, args.num_classes = data.x.shape[1], len(data.y.unique()) - 1
    if args.dataset == "pokec":
        args.num_classes = 1
    args.train_ratio, args.val_ratio = torch.tensor(
        [(data.y[data.train_mask] == 0).sum(), (data.y[data.train_mask] == 1).sum()]), \
                                       torch.tensor(
                                           [(data.y[data.val_mask] == 0).sum(), (data.y[data.val_mask] == 1).sum()])
    args.train_ratio, args.val_ratio = torch.max(args.train_ratio) / args.train_ratio, \
                                       torch.max(args.val_ratio) / args.val_ratio
    args.train_ratio, args.val_ratio = args.train_ratio[data.y[data.train_mask].long()], \
                                       args.val_ratio[data.y[data.val_mask].long()]



    acc, f1, auc_roc, parity, equality = run(data, args, data2)
    if args.ood == 3 or args.ood == 4:
        for i in range(len(args.strlist)):

            print("==========={}============".format(args.outid+args.strlist[i]))
            print('Acc: ', np.mean(acc.T[i]))
            print('auc_roc: ', np.mean(auc_roc.T[i]))
            print('parity: ', np.mean(parity.T[i]))
            print('equality: ', np.mean(equality.T[i]))
