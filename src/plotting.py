

# ###########################################################
# # Plotting

# Z = torch.load('Z.pth').numpy()
# dim_reducer = VAE()
# dim_reducer.load_state_dict(torch.load('sd.pth'))

# plt.ion(); plt.style.use('seaborn-pastel')

# plt.scatter(Z[np.isnan(y.cpu()) == 1, 0], Z[np.isnan(y.cpu()) == 1, 1], alpha=0.01)
# plt.scatter(Z[np.isnan(y.cpu()) == 0, 0], Z[np.isnan(y.cpu()) == 0, 1], alpha=0.25)

# fig, (axa, axb, axc) = plt.subplots(1, 3)
# axa.scatter(Z[:, 0], Z[:, 1], alpha=0.01)

# record = []
# while True:
#     idx = np.random.choice(len(segment_list), 1)[0]

#     start, end, f = segment_list[idx]
#     print(f'-----------------\n')
#     print(f'loading data: {segment_list[idx]}')
#     S, freq, t = get_spectrum_segment(start, end, f, extension=0.35)

#     mel_spec = melspectrogram(S=S, n_mels=n_mels)
#     mel_spec = standardize(Resize((n_mels, new_segm_len))(torch.tensor(mel_spec)[None, ...]))
#     mel_spec = binarize(mel_spec).to(device)

#     latent, _ = dim_reducer(mel_spec)
#     latent = latent.loc.cpu().detach()

#     axb.clear()
#     axb.imshow(S, aspect='auto', origin='lower', cmap=plt.cm.binary)

#     axc.imshow(mel_spec[0, 0].cpu(), aspect='auto', origin='lower')
#     plt.draw()

#     rect_start = int(0.35*S.shape[1] / ((end - start) + 2*0.35))
#     rect_end = S.shape[1] - rect_start
#     axb.add_patch(patches.Rectangle((rect_start, 0), rect_end - rect_start, len(freq), alpha=0.25))

#     label = input("Is this a call? y/n/m/.:\n")
#     col = dict(y='green', n='red', m='pink')[label]
#     axa.scatter(latent[0, 0], latent[0, 1], c=col)

#     record.append((
#         segment_list[idx],
#         label,
#     ))

# from matplotlib import offsetbox

# plt.figure()
# ax = plt.subplot(aspect='equal')

# plt.ylim(-5, 5)
# plt.xlim(-5, 5)
# plt.tight_layout()

# ax.scatter(Z[:, 0], Z[:, 1], lw=0, s=40, alpha=0.01)

# # idx_to_plot = np.random.choice(np.arange(n_data), 20, replace=False)
# shown_images = Z[[0], :]
# for i in range(len(Z)):
#     if np.square(Z[i] - shown_images).sum(axis=1).min() < 2:
#     # if i not in idx_to_plot:
#         continue
#     plt.scatter(Z[i, 0], Z[i, 1], c='black', alpha=0.7)
#     ax.add_artist(offsetbox.AnnotationBbox(
#         offsetbox.OffsetImage(X.cpu().numpy()[i, 0], cmap=plt.cm.autumn), Z[i, :]))
#     shown_images = np.r_[shown_images, Z[[i], :]]
# plt.xticks([]), plt.yticks([])
