function m = mape(testY, pred)

numerator = abs(bsxfun(@minus, pred, testY));
denominator = abs(testY);
m = mean(bsxfun(@rdivide, numerator, denominator))*100;
