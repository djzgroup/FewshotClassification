from model.utils import *

class PQFusion(nn.Module):
    def __init__(self, n_way, k_shot, q_query, K1=13, K2=2):
        super(PQFusion, self).__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        self.K1 = K1
        self.K2 = K2

        self.fc = nn.Linear(self.K1, 1)
        self.fc2 = nn.Linear(self.K2, 1)

        self.proto_weight = nn.Sequential(nn.Linear(4, 32),
                                          nn.ReLU(),
                                          nn.Linear(32, 4),
                                          nn.Softmax(-1))

        self.query_weight = nn.Sequential(nn.Linear(4, 32),
                                          nn.ReLU(),
                                          nn.Linear(32, 4),
                                          nn.Softmax(-1))


    def cif(self, proto, query):
        """
            Input:
                proto: (n_way, 1024)
                query: (n_way * q_query, 1024)
            Return:
                proto: (n_way, 1024)
                query: (n_way * q_query, 1024)
        """
        sim = cos_sim(proto, query)
        dist = -sim

        # == obtain fused proto ==
        index = torch.argsort(dist, -1)[:, :self.K1]
        selected_query = query[index.reshape(-1), :].reshape(self.n_way, self.K1, -1)
        squery_mean = selected_query.mean(1).unsqueeze(1)
        squery_max = selected_query.max(1)[0].unsqueeze(1)
        squery_mlp = self.fc(selected_query.transpose(1, 2)).transpose(1, 2)
        cat_proto = torch.cat([proto.unsqueeze(1), squery_mean, squery_max, squery_mlp], dim=1).permute(0, 2, 1)
        proto_weight = self.proto_weight(cat_proto)
        proto_feat = torch.sum(cat_proto * proto_weight, -1)

        # == obtain fused query ==
        dist = dist.permute(1, 0)
        index = torch.argsort(dist, -1)[:, :self.K2]
        selected_proto = proto[index.reshape(-1), :].reshape(self.n_way * self.q_query, self.K2, -1)
        sproto_mean = selected_proto.mean(1).unsqueeze(1)
        sproto_max = selected_proto.max(1)[0].unsqueeze(1)
        sproto_mlp = self.fc2(selected_proto.transpose(1, 2)).transpose(1, 2)
        cat_query = torch.cat([query.unsqueeze(1), sproto_mean, sproto_max, sproto_mlp], dim=1).permute(0, 2, 1)
        query_weight = self.query_weight(cat_query)
        query_feat = torch.sum(query_weight * cat_query, -1)

        return proto_feat, query_feat

    def cif_plus(self, proto, query):
        sim = cos_sim(proto, query)  # (n_way, q_query)
        upper_proto, upper_query = self.cif(proto, query)

        p2q_sim = F.softmax(sim, dim=-1)
        lower_proto = torch.matmul(p2q_sim, query)  # (n_way, emb_dim)

        sim = sim.transpose(0, 1)  # (q_query, n_way)
        q2p_sim = F.softmax(sim, dim=-1)
        lower_query = torch.matmul(q2p_sim, proto)  # (q_query, emb_dim)

        proto_feat = proto + upper_proto + lower_proto
        query_feat = query + upper_query + lower_query

        return proto_feat, query_feat


    def forward(self, proto, query):
        proto_feat, query_feat = self.cif_plus(proto, query)
        # proto_feat, query_feat = self.cif(proto, query)
        return proto_feat, query_feat

if __name__ == '__main__':
    n_way = 5
    k_shot = 5
    q_query = 10
    proto = torch.rand((5, 1024)).cuda()
    query = torch.rand((50, 1024)).cuda()
    model = PQFusion(n_way, k_shot, q_query).cuda()
    proto_feat, query_feat = model(proto, query)
    print(proto_feat.shape)
    print(query_feat.shape)
