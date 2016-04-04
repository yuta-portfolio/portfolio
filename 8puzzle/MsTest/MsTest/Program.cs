using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

/*	8パズル
 * 
 *  |￣￣￣|￣￣￣ |￣￣￣ | 
 *  |  1   |   2   |   3   |
 *  |      |       |       |
 *  |￣￣￣|￣￣￣ |￣￣￣ |
 *  |  4   |   5   |   6   |
 *  |      |       |       |
 *  |￣￣￣|￣￣￣ |￣￣￣ |
 *  |  7   |   8   |   0   |
 *  |      |       |       |
 *  ￣￣￣   ￣￣￣   ￣￣￣
 */

namespace MsTest
{
    class Program
    {
        static void Main(string[] args)
        {
            //パズルの初期状態を与える
			int[]puzzle = new int[]{ 1, 8, 0, 4, 3, 2, 5, 7, 6 };

            //Astar a=new Astar(3,3,puzzle);
            //a.Solve();
			Breadth b = new Breadth(3, 3, puzzle);
			b.Solve();
			Console.ReadLine();
        }
    }
}