function m = mae(testY, pred)

numerator = abs(bsxfun(@minus, pred, testY));
m = mean(numerator);
