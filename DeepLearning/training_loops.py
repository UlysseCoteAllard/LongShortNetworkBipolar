import time
import copy
import torch


def standard_training(model, cross_entropy_loss_for_class, optimizer_classifier, scheduler, dataloaders, device,
                      num_epochs=500, precision=1e-4, patience=30, patience_increase=30):
    since = time.time()
    best_loss = float('inf')
    best_accuracy = 0.
    best_state = {'epoch': 0, 'state_dict': copy.deepcopy(model.state_dict()),
                  'optimizer': optimizer_classifier.state_dict(), 'scheduler': scheduler.state_dict()}

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        'Training'
        # Each epoch has a training and validation phase
        for phase in ['train', 'val', "test"]:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.
            running_corrects = 0
            total = 0

            for i, data in enumerate(dataloaders[phase], 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer_classifier.zero_grad()
                if phase == 'train':
                    if len(inputs) > 1:  # Make sure the batch size is > 0 because we use batch norm
                        # forward
                        outputs = model(inputs)
                        _, predictions = torch.max(outputs.data, -1)
                        loss = cross_entropy_loss_for_class(outputs, labels)
                        loss.backward()
                        optimizer_classifier.step()
                        loss = loss.item()
                        # statistics
                        running_loss += loss
                        running_corrects += torch.sum(predictions == labels.data)
                        total += labels.size(0)
                else:
                    with torch.no_grad():
                        # forward
                        outputs = model(inputs)
                        _, predictions = torch.max(outputs.data, 1)

                        loss = cross_entropy_loss_for_class(outputs, labels)
                        loss = loss.item()
                        # statistics
                        running_loss += loss
                        running_corrects += torch.sum(predictions == labels.data)
                        total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.item() / total
            print('{} Loss: {:.8f} Acc: {:.8}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_loss + precision < best_loss:
                    print("New best validation loss:", epoch_loss, " Corresponding validation accuracy: ", epoch_acc)
                    best_accuracy = epoch_acc
                    best_loss = epoch_loss
                    best_state = {'epoch': epoch + 1, 'state_dict': copy.deepcopy(model.state_dict()),
                                  'optimizer': optimizer_classifier.state_dict(), 'scheduler': scheduler.state_dict()}
                    patience = patience_increase + epoch
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - epoch_start))
        torch.cuda.empty_cache()
        if epoch > patience:
            break

    print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:4f}'.format(best_accuracy))
    return best_state
