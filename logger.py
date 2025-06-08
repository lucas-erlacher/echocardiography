# this class standardizes logging across all training scripts
# NOTE: I think the reason why I get nonzero CE even if confusion matrix is 100% perfect is because confusion matrix displays argmaxes of predicted
# distrib BUT cross entropy considers whole distribution i.e. unless we return perfect 1-hot distribs we will not see a CE of exactly zero. 

import torch
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import PIL.Image as Image
from torchvision import transforms
import matplotlib
from reducer import Reducer

class Logger:
    # any items logged by this logger will be grouped together by prefix
    # i.e. if e.g. prefix="extractor" all logged extractor items will be grouped together in tensorboard (which makes the board much easier to navigate). 
    # 
    # enable aggregate_views if second dimension of data_tensor that is passed to the methods of this class 
    # iterates over multiple views in which case we want to aggregate all views togehther under one collapsable heading.
    #
    # use optional parameter plot_dpi to control the image quality of the plots logged by this logger (which helps to keep the footprint of the tensorboards reasonable)
    # 
    # the latent space visualizations are quite computationally epxensive. 
    # set visualization_limit to reduce that complexity OR to make different plots have same number of entries (e.g. to make visualizations on test and eval data more comparable). 
    def __init__(self, seed, writer, prefix, aggregate_views=False, plot_dpi=100, visualization_limit=None):
        self.step = 0
        self.writer = writer
        self.prefix = prefix 
        self.aggregate_views = aggregate_views
        self.plot_dpi = plot_dpi
        self.visualization_limit = visualization_limit
        self.seed = seed

    def increment_step(self):
        self.step += 1

    # loss_info is assumed to be a batch of items
    # info types is a list of textual descriptions of each type of loss info present in loss_info
    def log_loss_info(self, info_types, loss_info):
        if self.aggregate_views:  # if data for multiple views has been passed to this method we first need to iterate over the views
            for i in range(len(loss_info)):  # iterate over views
                for info_idx, info_type in enumerate(info_types): 
                    curr_view = loss_info[i]
                    curr_info = curr_view[info_idx]
                    self.writer.add_scalar(self.prefix + ' - ' + info_type + '/view ' + str(i + 1), self.__combination_strategy(curr_info), self.step)
        else:  
            for info_idx, info_type in enumerate(info_types): 
                curr_info = loss_info[info_idx]
                self.writer.add_scalar(self.prefix + ' - ' + info_type, self.__combination_strategy(curr_info), self.step)

    # this method combines an item from each batch i.e. it assumes that batch_dim has already been reduced over
    # currently we are simply averaging but this can be changed in the future (hence I created this method s.t. I can easily change the combination strategy)
    def __combination_strategy(self, items):
        combined = torch.mean(items)
        return combined

    # image_tensor is assumed to NOT be a batch of images
    # card-title can be used to insert additional information into the tensorboard-card title (e.g. "input" or "recon")
    def log_image(self, image_tensor, card_title=""):
        if card_title != "": card_title = "/" + card_title  # add slash to card_title if present s.t. tb will struture the content properly
        if self.aggregate_views: 
            for i, input in enumerate(image_tensor): 
                if len(input.shape) == 5:  # if we are in block_mode log first and last image downsampled 2x and merged side by side
                    input_1 = torch.nn.functional.interpolate(input[:, :, 0, :, :], scale_factor=0.5, mode='bilinear', align_corners=False)
                    input_2 = torch.nn.functional.interpolate(input[:, :, -1, :, :], scale_factor=0.5, mode='bilinear', align_corners=False)
                    input = torch.cat((input_1, input_2), dim=-1)
                self.writer.add_images(self.prefix + " - images" + '/view ' + str(i + 1) + card_title, input, self.step)
        else:
            self.writer.add_image(self.prefix + " - images" + card_title, image_tensor, self.step)

    # log histogram (over model layers) where each bar is the sum (over all parameters in a given layer) of the norms of the gradients
    # this shows which layers are altered by how much (which can inform architectural decisions)
    def log_gradients(self, model):
        if self.aggregate_views:  # in this case we expect the model parameter to be a list of models
            for model_idx, m in enumerate(model):
                hist = self.__get_grad_histogram(m)
                self.writer.add_image(self.prefix + ' - ' + 'layer-wise gradient norm sums' + '/view ' + str(model_idx + 1), hist, self.step)  
        else:
            hist = self.__get_grad_histogram(model)
            self.writer.add_image(self.prefix + ' - ' + 'layer-wise gradient norm sums', hist, self.step)  

   
    # model can either be one nn.Module or a list of nn.Modules (whose histograms will then all be displayed in the same image)
    def __get_grad_histogram(self, models):
        if not isinstance(models, list):
            models = [models]  # if ony one model was passed to method wrap it in a list in order for the following code to work
        images = []
        
        for model in models:
            layer_names = []
            norms_per_layer = []

            for name, module in model.named_modules():
                if name == "":
                    continue  # do not display root module in hist

                layer_names.append(name)
                norm_sum = sum(param.grad.norm().item() for param in module.parameters() if param.grad is not None)
                norms_per_layer.append(norm_sum)

            # histogram for the current model
            plt.figure(figsize=(12, 6))
            plt.bar(layer_names, norms_per_layer, color='#A0C4FF')
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('layer names')
            plt.ylabel('sum of gradient norms')
            plt.title(model.__class__.__name__)
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=self.plot_dpi)
            plt.close()
            buf.seek(0)
            image = Image.open(buf)
            images.append(image)

        # combine images vertically
        widths, heights = zip(*(i.size for i in images))
        total_height = sum(heights)
        max_width = max(widths)
        combined_image = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for img in images:
            combined_image.paste(img, (0, y_offset))
            y_offset += img.size[1]
        
        return transforms.ToTensor()(combined_image)

    # visualize the latent space structure (using t-SNE and PCA for dimensionality reduction)
    # t-sne plots are highly dependent on perplexity i.e. create plots for multiple perplexity values s.t. we can compare them afterwards
    def visualize_latent_space(self, latent_representations, labels):
        # convert nan to -1 for labels
        labels = list(map(lambda x: x.item(), labels))  # Convert labels from tensors to numbers 
        labels = list(map(lambda x: -1 if np.isnan(x) else x, labels))  # TSNE/UMAP can't handle nans

        plt.figure(figsize=(20, 12))

        perplexities = [5, 30, 50]  # t-SNE perplexity values (which are recommended to stay in [5, 50])
        n_neighbors_values = [15, 30, 60]  # don't go too high on this parameter as it is not supposed to exceed the number of latent_representations (typically 160)
        color_map = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # distinct colors for the classes

        # apply visualization_fraction if enabled by user
        if self.visualization_limit != None:
            latent_representations = latent_representations[:self.visualization_limit, :]
            labels = labels[:self.visualization_limit]

        red = Reducer(self.seed, 2)  # reduce latent codes down to 2 dims

        # tsne
        for i, perplexity in enumerate(perplexities):
            reduced_latent_tsne = red.tsne(latent_representations, perplexity=perplexity)
            # create subplot for each t-SNE plot
            plt.subplot(3, len(perplexities), i + 1)
            scatter_tsne = plt.scatter(reduced_latent_tsne[:, 0], reduced_latent_tsne[:, 1], 
                                    c=labels, cmap=matplotlib.colors.ListedColormap(color_map), 
                                    alpha=0.7, s=50)
            unique_labels = np.unique(labels)
            plt.legend(*scatter_tsne.legend_elements(), title="classes", loc='upper right')
            plt.ylabel('t-SNE component 1')
            plt.xlabel('t-SNE component 2')
            plt.title(f't-SNE (perplexity={perplexity})')

        # umap
        for j, n in enumerate(n_neighbors_values):
            reduced_latent_umap = red.umap(latent_representations, num_neighbors=n)
            # create subplot for each UMAP plot
            plt.subplot(3, len(perplexities), len(perplexities) + j + 1)
            scatter_umap = plt.scatter(reduced_latent_umap[:, 0], reduced_latent_umap[:, 1], 
                                    c=labels, cmap=matplotlib.colors.ListedColormap(color_map), 
                                    alpha=0.7, s=50)
            plt.legend(*scatter_umap.legend_elements(), title="classes", loc='upper right')
            plt.ylabel('UMAP component 1')
            plt.xlabel('UMAP component 2')
            plt.title(f'UMAP (n_neighbors={n})')

        # pca
        reduced_latent_pca = red.pca(latent_representations)
        # create subplot for pca
        plt.subplot(3, len(perplexities), 2 * len(perplexities) + 1)
        scatter_pca = plt.scatter(reduced_latent_pca[:, 0], reduced_latent_pca[:, 1], 
                                c=labels, cmap=matplotlib.colors.ListedColormap(color_map), 
                                alpha=0.7, s=50)
        plt.legend(*scatter_pca.legend_elements(), title="classes", loc='upper right')
        plt.xlabel('PCA component 1')
        plt.ylabel('PCA component 2')
        plt.title('PCA')

        plt.tight_layout()

        # save plots as bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=self.plot_dpi)
        plt.close()
        buf.seek(0)
        
        # log the combined image to tb
        image = Image.open(buf)
        self.writer.add_image(self.prefix + ' - latent space visualization', transforms.ToTensor()(image), self.step)

    # create and log confusion matrix to tb
    def log_confusion_matrix(self, true_labels, predicted_labels):
        zipped = list(zip(true_labels, predicted_labels))
        res_matrix = torch.zeros((3, 3), dtype=int)  
        for tup in zipped:
            res_matrix[int(tup[0])][int(tup[1])] += 1
        # normalize counts to percentages
        row_sums = res_matrix.sum(dim=1, keepdim=True)
        norm_matrix = res_matrix.float() / row_sums  
        # create heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(norm_matrix.numpy(), interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(3)
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        # insert percentages into cells
        thresh = norm_matrix.max() / 2.  
        for i, j in np.ndindex(norm_matrix.shape):
            plt.text(j, i, f'{norm_matrix[i, j].item():.2f}',
                    horizontalalignment="center",
                    color="white" if norm_matrix[i, j] > thresh else "black")
        plt.ylabel('true Label')
        plt.xlabel('predicted Label')
        # by using bytesIO we avoid having to save an image to disk
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=self.plot_dpi)
        buf.seek(0) 
        self.writer.add_image(self.prefix + ' - confusion matrix', plt.imread(buf), global_step=self.step, dataformats='HWC')
        plt.close()  # if we don't close we might run into memory leaks