import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import DGCNN_cls
from model.ICM import Enrich_proto
from model.IFM import PQFusion

class ProtoNet(nn.Module):
    def __init__(self, args):
        super(ProtoNet, self).__init__()
        self.encoder = DGCNN_cls(args)
        self.n_classes = args.n_way
        self.n_support = args.k_shot
        self.n_query = args.q_query
        self.in_channel = args.in_channel
        self.emb_dims = args.emb_dims
        self.n_points = args.pc_npts


        self.use_sup = args.use_sup
        if self.use_sup and self.n_support == 5:
            self.sup_model = Enrich_proto(self.n_classes, self.n_support, self.n_query)

        self.use_pqf = args.use_pqf
        if self.use_pqf:
            self.pqf_model = PQFusion(self.n_classes, self.n_support, self.n_query)


    def forward(self, support, query, query_y):
        support = support.view(self.n_classes * self.n_support, self.n_points, self.in_channel)
        support_feat = self.encoder(support.transpose(2, 1))
        proto_feat = support_feat.view(self.n_classes, self.n_support, -1).mean(1) # (n_way, 1024)

        query = query.view(self.n_classes * self.n_query, self.n_points, self.in_channel)
        query_feat = self.encoder(query.transpose(2, 1)) # (n_way * q_query, 1024)

        if self.use_sup and self.n_support == 5:
            proto_feat = self.sup_model(proto_feat, support_feat)

        if self.use_pqf:
            proto_feat, query_feat = self.pqf_model(proto_feat, query_feat)

        dist = torch.cdist(query_feat, proto_feat)
        pred = F.softmax(-dist, 1)
        pred_log = F.log_softmax(-dist, 1).view(self.n_classes, self.n_query, -1)
        loss = F.nll_loss(pred_log.transpose(1, 2), query_y.long())
        return pred, loss







