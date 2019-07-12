
在线求导工具:</br>
http://www.nicetool.net/app/derivative.html </br>

输入(x+5)^2 后得出倒数就是2 * (x+5)</br>

https://www.derivative-calculator.net/


https://www.shuxuele.com/calculus/derivatives-introduction.html</br>


# 导数入门
全是与坡度有关！</br>
坡度 = y的改变 / x的改变</br>
<img border="0" src="images/derivative/slope.svg"/></br>
我们可以求两点之间的 **平均** 坡度</br>
<img border="0" src="images/derivative/slope-average.svg"/></br>
但我们怎样求在一点的坡度？</br>
没有什么可以测量的！</br>
<img border="0" src="images/derivative/slope-0-0.svg"/></br>
但是，在导数里，我们可以用一个很小的差……</br>
……然后把它**缩小到零**。</br>
坡度 = delta y / delta x</br>

**求个导数**！</br>
求函数 y = f(x) 的导数，我们用坡度的公式：</br>

坡度 = Y的改变 / X的改变 = Δy / Δx </br>

我们看到（如图）:</br>
x 从 	  	x 	变到 	x+Δx</br>
y 从 	  	f(x) 	变到 	f(x+Δx)</br>
<img border="0" src="images/derivative/slope-dy-dx2.gif"/></br>
按照这步骤去做：</br>


    代入这个坡度公式： Δy / Δx = f(x+Δx) − f(x) / Δx
    尽量简化
    把 Δx 缩小到零。

像这样：</br>

例子：函数 f(x) = x^2</br>
我们知道 f(x) = x^2，也可以计算 f(x+Δx) ：</br>
开始：f(x+Δx) = (x+Δx)^2</br>
(x + Δx)^2:f(x+Δx) = x^2 + 2x Δx + (Δx)^2</br>
坡度公式是： 	(f(x+Δx) − f(x)) / Δx </br>
代入 f(x+Δx) 和 f(x)： 	(x^2 + 2x Δx + (Δx)^2 − x^2) / Δx </br>
简化 (x2 and −x2 约去）： 	(2x Δx + (Δx)^2) / Δx </br>
再简化（除以 Δx）： 	  = 2x + Δx</br>
当 Δx 趋近 0时，我们得到： 	= 2x</br>
结果：x^2 的导数是 2x</br>
我们写 dx，而不写 "Δx 趋近 0"，所以 "的导数" 通常是写成 d/dx</br>
d/dx * x2 = 2x</br>
"x^2 的导数等于 2x"</br>
或 "x^2 的 d dx 等于 2x"</br>
d/dx * x2 = 2x 的意思是什么？</br>
意思是，对于函数 x^2，在任何一点的坡度或 "变化率" 是 2x。</br>
所以当 x=2，坡度是 2x = 4，如图所示：</br>
或当 x=5，坡度是 2x = 10，以此类推。</br>
<img border="0" src="images/derivative/slope-x2-2.svg"/></br>
注意：f’(x) 也是 "的导数" 的另一个写法：</br>
f’(x) = 2x</br>
"f(x) 的导数等于 2x"</br>



