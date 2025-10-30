from models.osnet import OSNetFeatureExtractor, compute_pairwise_similarity


def main():
    extractor = OSNetFeatureExtractor(
        model_name='osnet_ain_x1_0',
        model_path='models/osnet/checkpoints/osnet_ain_x1_0_msmt17.pth',  # update path
        device='cuda'
    )

    image_list = [
        '/data/reid/reid_master/tracklets/1F/track_0004/track0004_frame001650.jpg',
        '/data/reid/reid_master/tracklets/1F/track_0004/track0004_frame001615.jpg',
        '/data/reid/reid_master/tracklets/1F/track_0006/track0006_frame003320.jpg'
    ]

    feats = extractor(image_list)
    print('feats:', feats.shape)

    cos = compute_pairwise_similarity(feats, metric='cosine')
    print('cosine similarity (NxN):')
    print(cos.cpu().numpy())

    euc = compute_pairwise_similarity(feats, metric='euclidean')
    print('euclidean distance (NxN):')
    print(euc.cpu().numpy())


if __name__ == '__main__':
    main()