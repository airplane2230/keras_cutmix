import numpy as np

class CutMixGenerator():
    def __init__(self, df, generator, TRAIN_IMG_PATH, alpha, num_classes, batch_size, img_size, cutmix_in_train=True):
        self.batch_index = 0
        self.alpha = alpha
        self.batch_size = 32
        self.num_classes = num_classes
        self.TRAIN_IMG_PATH = TRAIN_IMG_PATH

        self.generator = generator.flow_from_dataframe(
            dataframe=df,
            directory=self.TRAIN_IMG_PATH,
            x_col='img_file',
            y_col='class',
            target_size=img_size,
            color_mode='rgb',
            class_mode='categorical', # change your's
            batch_size=batch_size,
            shuffle=True
        )
        self.n = self.generator.samples
        self.cutmix_in_train = cutmix_in_train

    @property
    def class_indices(self):
        return self.generator.class_indices

    @property
    def samples(self):
        return self.generator.samples

    def rand_bbox(self, size, lam):
        W = size[1]
        H = size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix(self, batch, alpha, batch_size):
        data, targets = batch

        indices = np.random.permutation(batch_size)
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]

        lam = np.random.beta(alpha, alpha)
        lam = np.array(lam)

        image_w, image_h = data.shape[1], data.shape[2]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(data.shape, lam)

        data[:, bbx1:bbx2, bby1:bby2, :] = shuffled_data[:, bbx1:bbx2, bby1:bby2, :]
        lam = np.repeat(lam, data.shape[0])

        return data, targets, shuffled_targets, lam

    def reset_index(self):
        self.generator._set_index_array()

    def on_epoch_end(self):
        self.reset_index()

    def get_steps_per_epoch(self):
        # 떨어지는 경우와 그렇지 않은 경우를 구분해서 총 step수를 반환해 준다.
        quotient, remainder = divmod(self.n, self.batch_size)
        return (quotient + 1) if remainder else quotient

    def __next__(self):
        X, y = self.generator.next()

        if self.batch_index == 0:
            self.reset_index()

        # current_index는 전체 샘플 수 중 batch step 수
        current_index = (self.batch_index * self.batch_size) % self.n

        # batch step이 전체 샘플 수를 넘겼다면 batch_index + 1
        if self.n > current_index + self.batch_size:
            self.batch_index += 1
        else:
            self.batch_index = 0

        # 아무 조건이 없다면 배치사이즈만큼 반환하므로
        reshape_size = self.batch_size
        # 밑의 조건문은 배치 사이즈보다 적은 수의 샘플이 남은 경우에 reshape_size를
        # 그에 맞게 조절해주는 코드
        if current_index == (self.get_steps_per_epoch() - 1) * self.batch_size:
            reshape_size = self.n - ((self.get_steps_per_epoch() - 1) * self.batch_size)

        if self.cutmix_in_train:
            # data, (targets, shuffled_targets, lam)
            data, targets, shuffled_targets, lam = self.cutmix((X, y), self.alpha, reshape_size)
            return [data, targets, shuffled_targets, lam], targets
        else:
            return [X, np.zeros((reshape_size, 196)), np.zeros((reshape_size, 1))], y

    def __iter__(self):
        while (True):
            yield next(self)