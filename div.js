
// const tf = require("../tf.js");

/**
 * @method tf.div (a, b)
 * 1. a에 대응하는 b의 값으로 나눈다..
 * 2. 브로드캐스팅(broadcating)을 지원한다.
 * - 단, a 또는 b에 shape가 하나일 때이다.
 * @param {tf.Tensor} a,
 * @param {tf.Tensor} b,
 * 1. Must have the same dtype as a.
 */
const tsOne = tf.tensor1d([5, 15, 30]);
const tsTwo = tf.tensor1d([2, 3, 4]);

tsOne.div(tsTwo).print();
web.log(tsOne.div(tsTwo));

// [2.5, 5, 7.5]
//----------------------------

/**
 * 1. 1을 1로 나누고 4를 2로 나눈다.
 * 2. 9/3, 16/4를 한다
 */
const tsThree = tf.tensor([[1, 4], [9, 16]])
const tsFour = tf.tensor([[1, 2], [3, 4]]);

tsThree.div(tsFour).print();
web.logTensor(tsThree.div(tsFour));

// [[1, 2], [3, 4]]
//----------------------------

/**
 * 1. 브로드케스팅(broadcast) 처리
 * 2. [1, 2, 3, 4]의 각 shape에 5를 곱한다
 */
const tsFive = tf.tensor1d([3, 6, 12, 15]);
const tsSix = tf.scalar(3);

tsFive.div(tsSix).print();
web.logTensor(tsFive.div(tsSix));

// [1, 2, 4, 5]
//------------------------

