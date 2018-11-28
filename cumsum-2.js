
//const tf = require('../../tf.js');

const tsOne = tf.tensor1d([1, 3, 5, 7, 11]);
tf.cumsum(tsOne, 0, true).print();
web.log(tf.cumsum(tsOne, 0, true));
//---------------

tf.cumsum(tsOne, 0, false, true).print();
web.log(tf.cumsum(tsOne, 0, false, true));

// 27, 27-1=26, 27-(1+3)=23, 27-(1+3+5)=18, 27-(1+3+5+7)=1