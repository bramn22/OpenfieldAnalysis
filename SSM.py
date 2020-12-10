import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2

""" Statistical Shape Model"""
class SSM:

    def fit(self, df, bodyparts, n_components=3):
        df = df.copy()
        """ Requires a standardized (PPS) pose dataset as DataFrame. """
        parts_matrix = np.zeros((len(df), len(bodyparts) * 2))
        for j, bodypart in enumerate(bodyparts):
            parts_matrix[:, j * 2:j * 2 + 2] = df[bodypart][['x', 'y']]
        print(parts_matrix.shape)

        self.scaler = StandardScaler()
        parts_matrix_scaled = self.scaler.fit_transform(parts_matrix)

        self.pca = PCA(n_components=n_components)
        self.pca.fit(parts_matrix_scaled)
        self.components = self.pca.components_
        plt.figure()
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.show()

    def transform(self, df, bodyparts):
        df = df.copy()
        parts_matrix = np.zeros((len(df), len(bodyparts) * 2))
        for j, bodypart in enumerate(bodyparts):
            parts_matrix[:, j * 2:j * 2 + 2] = df[bodypart][['x', 'y']]
        print(parts_matrix.shape)
        parts_matrix_scaled = self.scaler.transform(parts_matrix)
        pca_embeddings = self.pca.transform(parts_matrix_scaled)
        for i, pc_embeddings in enumerate(pca_embeddings.T):
            df['SSM', f'pc{i}'] = pc_embeddings
        return df

    def inverse_transform_single(self, shape_params):
        return self.scaler.inverse_transform(shape_params @ self.components)

    def correct_outliers(self):
        pass


    # def create_pcs_video(self, output_path):
    #     fps = 30
    #     width, height, output, video = None, None, None, None
    #
    #     t_start, t_end = 0, 2
    #     ts = np.linspace(t_start, t_end, 200 * t_end)
    #     vs = np.sin(2 * np.math.pi * ts * (6 ** (1 / 2)))
    #     for v in vs:
    #         img = self._get_plot(v)
    #
    #         if width is None:
    #             width = img.shape[1]
    #             height = img.shape[0]
    #             video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (int(width), int(height)))
    #
    #         video.write(img)
    #
    #     video.release()
    #     cv2.destroyAllWindows()
    #
    # def _get_plot(self, v):
    #     fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    #
    #     recon_1 = scaler.inverse_transform(np.array([v, 0, 0]) @ pca.components_)
    #     recon_2 = scaler.inverse_transform(np.array([0, v, 0]) @ pca.components_)
    #     recon_3 = scaler.inverse_transform(np.array([0, 0, v]) @ pca.components_)
    #
    #     for j, bodypart in enumerate(bodyparts_body):
    #         axs[0].scatter(recon_1[j * 2], recon_1[j * 2 + 1], c=[colors[j]], s=80)
    #         axs[1].scatter(recon_2[j * 2], recon_2[j * 2 + 1], c=[colors[j]], s=80)
    #         axs[2].scatter(recon_3[j * 2], recon_3[j * 2 + 1], c=[colors[j]], s=80)
    #     for ax in axs:
    #         ax.set_aspect('equal', adjustable='box')
    #         ax.set(xlim=[-35, 35], ylim=[-35, 35])
    #     fig.tight_layout()
    #     fig.canvas.draw()
    #     img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #     plt.close()
    #     img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #     # img is rgb, convert to opencv's default bgr
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     return img