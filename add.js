
//const tf = require("../tf.js");

/**
 * @method tf.add (a, b)
 * 1. a의 값 엘리먼트 값에 b에서 대응하는 엘리먼트 값을 더한다.
 * - 대응한다는 것은 행렬 계산의 대응과 같다.
 * - a 또는 b 텐서의 값이 변경되지 않는다.
 * 2. a의 1에 대응하는 것이 b의 10이고, add를 하므로 11이 된다.
 * 3. a의 2에 대응하는 것이 b의 20이고, add를 하므로 22가 된다.
 * 4. a의 3에 대응하는 것이 b의 30이고, add를 하므로 33이 된다.
 * 5. tf.add(a, b) 형태로도 작성할 수 있다.
 * 2. 브로드캐스팅(broadcating)을 지원한다.
 * - 단, a 또는 b에 shape가 하나일 때이다.
 * @param {tf.Tensor} a
 * 1. The first tf.Tensor to add.
 * @param {tf.Tensor} b,
 * 1. The second tf.Tensor to add.
 * - Must have the same type as a.
 * @return tf.Tensor
 */
const tsOne = tf.tensor1d([1, 2, 3]);
const tsTwo = tf.tensor1d([5, 6, 7]);

tsOne.add(tsTwo).print();
web.log(tsOne.add(tsTwo));
// [6, 8, 10]
//----------------

tsOne.print();
web.log(tsOne);
// [1, 2, 3]
//----------------

const tsResult = tf.add(tsOne, tsTwo);
tsResult.print();
web.log(tsResult);
// [6, 8, 10]
//----------------

const tsThree = tf.tensor([[1, 2], [3, 4]])
const tsFour = tf.tensor([[5, 6], [7, 8]]);

tsThree.add(tsFour).print();
web.log(tsThree.add(tsFour));
// [[6, 8], [10, 12]]
//----------------
/**
 * 1. 브로드케스팅(broadcast) 처리
 * - a의 5에 b의 각 엘리먼트 값을 더해
 * - 4개의 shape를 가진 벡터를 반환한다.
 * 2. a에 5 하나만 작성한 반면 b에는 5개를 작성했다.
 * - 행렬의 더하기를 하려면 앞의 코드와 같이
 * - tf.tensor1d([5, 5, 5, 5, 5]); 형태로 작성해야 한다.
 * - 그런데 output 결과에서 보듯이 tf.scalar(5);에서 5가
 * - tf.tensor1d([10, 20, 30, 40]);의 각 엘리먼트에 더해진다.
 * - 이를 브로드케스팅이라고 한다.
 * 2. tf.add(a, b) 형태로도 작성할 수 있다.
 */
const tsFive = tf.scalar(7);
const tsSix = tf.tensor1d([3, 5, 7]);

tsFive.add(tsSix).print();
web.log(tsFive.add(tsSix));
// [15, 25, 35, 45]
//-------------------------


/**
 * 1. tf.tensor1d([1, 2])와 같이 shape를 2개 작성하고
 * - tf.tensor1d([10, 20, 30, 40]);와 같이 4개를 작성한 상태에서
 * 2. e.add(f)를 하면 브로드케스팅을 할 수 없다고 에러가 발생한다.
 * 3. 따라서 브로드케스팅을 하려면 하나만 작성해야 한다.
 */
// const g = tf.tensor1d([1, 2]);
// const h = tf.tensor1d([10, 20, 30, 40]);
// g.add(h).print();
//-------------------------

/**
 * 수학 행렬의 더하기/빼기 계산 방법
 */
/*
1과 9를 더하고, 2와 8을 더하고, 3과 7을 더한다.
(          (          (               (
 1 2 3      9 8 7      1+9 2+8 3+7     10 10 10
 4 5 6   +  1 2 3  =   4+1 5+2 6+3  =  5  7  9
 7 8 9      4 5 6      7+4 8+5 9+6     11 13 15
)          )          )               )
*/
//----------------------

/**
 * 1. 수학 행렬의 곱하기는 더하기와 다른 경우가 있다.
 * 2. 이 폴더의 mul.js에 작성하였다.
 */
