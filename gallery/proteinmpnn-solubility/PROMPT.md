# ProteinMPNN for soluble designs

What the scientist sent:

> Current ProteinMPNN tends to design insoluble protein sequences that aggregate or are unable
> to express. I wanna develop a version of ProteinMPNN that can design soluble proteins; given
> the same backbone, it should put fewer water-hating residues on the surface, while folding into
> the backbone we asked for.
>
> I'll give u the original ProteinMPNN train data (the data link), here below, a CSV listing the
> PDB entries that should be left out when you want soluble-only training: membrane proteins and
> the like.
>
> For the evaluation, use ESMFold to fold designed sequences back and compare them with input
> backbones, and see the self-consistency TM-score and more soluble residues on the surface. Use
> the H200 GPUs on the cluster.
>
> Can you build this with The-ML-Guy?
