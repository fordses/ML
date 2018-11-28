
//const tf = require('../../tf.js');

const tsOne = tf.tensor1d([1, 3, 5, 7]);
tf.cumsum(tsOne).print();

web.log(tf.cumsum(tsOne));
//---------------

const tsTwo = tf.tensor2d([1, 3, 5, 7, 9, 11], [2, 3]);
tf.cumsum(tsTwo).print();
tf.cumsum(tsTwo, 1).print();

web.log(tsTwo.cumsum());
web.log(tsTwo.cumsum(1));



