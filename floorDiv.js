
// const tf = require("../../tf.js");

/**
 * @method tf.floorDiv (a, b)
 * 1. 두 개의 Tensor 엘리먼트 각각을 나누고 나머지를 버린다.
 * 2. 브로드캐스팅(broadcating)을 지원한다.
 * - 단, a 또는 b에 shape가 하나일 때이다.
 * @param {tf.Tensor} a,
 * @param {tf.Tensor} b,
 * 1. Must have the same dtype as a.
 */
/**
 * 1. a에 대응하는 b의 값으로 나눈다.
 * 2. tf.div(a, b) 형태로도 작성할 수 있다.
 */
const tsOne = tf.tensor1d([5, 7, 15]);
const tsTwo = tf.tensor1d([2, 3, 4]);

tsOne.floorDiv(tsTwo).print();
web.log(tsOne.floorDiv(tsTwo));

// [2, 2, 3]
//----------------------------

/**
 * 1. 1을 1로 나누고 4를 2로 나눈다.
 * 2. 9/3, 16/4를 한다
 */
const tsThree = tf.tensor([[5, 7], [13, 16]])
const tsFour = tf.tensor([[2, 3], [4, 5]]);

tsThree.floorDiv(tsFour).print();
web.log(tsThree.floorDiv(tsFour));

// [[2, 2], [3, 3]]
//----------------------------


/**
 * 1. 브로드케스팅(broadcast) 처리
 * 2. [1, 2, 3, 4]의 각 shape에 5를 곱한다
 */
const tsFive = tf.tensor1d([5, 7, 15]);
const tsSix = tf.scalar(3);

tsFive.floorDiv(tsSix).print();
web.log(tsFive.floorDiv(tsSix));

// [1, 2, 5]
//------------------------

