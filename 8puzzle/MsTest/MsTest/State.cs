/*State.cs*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MsTest
{
	public class State
	{
		public State preState;//一つ前の状態
		public int preMove;//一つ前のパネルの移動

		//パネルの移動(0の移動と逆になる) 0.→ 1.↓ 2.← 3.↑
		static int[] vy = new int[] { 0, 1, 0, -1 };
		static int[] vx = new int[] { 1, 0, -1, 0 };
		public static String[] vName = new String[] { "left", "up", "right", "down" };

		int[,] nums;//パズルの状態
		int zeroY, zeroX;//空欄の存在する位置
		int hash;//状態に対応づける数値（初期値-1）

		public int cost, g, cnm;
		public int depth, h;

		//移動可能な状態をLinkedListオブジェクトに入れて返す(OpenList)
		public LinkedList<State> nextMove()
		{
			LinkedList<State> ret = new LinkedList<State>();
			//空欄の動かし方は上下左右の4通り
			for (int i = 0; i < 4; i++)
			{
				int nextZeroY = zeroY + vy[i];
				int nextZeroX = zeroX + vx[i];
				if (nextZeroX < 0 || nextZeroX >= 3) continue;
				if (nextZeroY < 0 || nextZeroY >= 3) continue;
				int[,] nextNums = new int[3, 3];
				for (int y = 0; y < 3; y++)
				{
					for (int x = 0; x < 3; x++)
					{
						nextNums[y, x] = nums[y, x];
					}
				}
				nextNums[nextZeroY, nextZeroX] = 0;
				nextNums[zeroY, zeroX] = nums[nextZeroY, nextZeroX];
				ret.AddLast(new State(this, i, nextNums, nextZeroY, nextZeroX, depth));

			}
			return ret;
		}

		//Stateコンストラクタ
		public State(State _preState, int _preMove, int[,] _nums, int _zeroY, int _zeroX, int _depth)
		{
			preState = _preState;
			preMove = _preMove;
			nums = _nums;
			zeroY = _zeroY;
			zeroX = _zeroX;
			hash = -1;
			depth = _depth + 1;
			setCost();
			if (preState != null)
			{

			}
		}

		//パズルの状態を一意に数値に対応させる
		public int hashCode()
		{
			if (hash != -1) return hash;
			hash = 0;
			for (int y = 0; y < 3; y++)
			{
				for (int x = 0; x < 3; x++)
				{
					hash *= 10;
					hash += nums[y, x];
				}
			}
			return hash;
		}

		//ヒューリスティック値を求める
		public int getHeuristic()
		{
			int lx, ly;
			int ans = 0;
			int sum = 0;
			for (int y = 0; y < 3; y++)
			{
				for (int x = 0; x < 3; x++)
				{
					++ans;

					if (nums[y, x] != 0)
					{

						ly = Math.Abs((int)Math.Floor((((double)nums[y, x] - 1) / 3) - (int)Math.Floor((double)(ans - 1) / 3)));
						lx = Math.Abs((nums[y, x] - 1) % 3 - (ans - 1) % 3);

					}
					else
					{
						continue;
					}

					sum += (lx + ly);
				}
			}

			return sum;
		}

		//コストを設定する
		public void setCost()
		{
			h = getHeuristic();
			g = depth;
			cost = g + h;
		}
		public void setCost(int _g, int _h)
		{
			g = _g;
			h = _h;
			cost = g + h;
		}

		//現在の状態を表示
		public void showState()
		{
			Console.WriteLine("[" + depth + "]g*(n):{0}  h*(n):{1}  f*(n)={2}:", g, h, cost);
			_showState();
		}
		private void _showState()
		{
			Console.WriteLine("|￣￣￣|￣￣￣|￣￣￣|");
			Console.WriteLine("|  {0}   |  {1}   |  {2}   |", nums[0, 0], nums[0, 1], nums[0, 2]);
			Console.WriteLine("|      |      |      |");
			Console.WriteLine("|￣￣￣|￣￣￣|￣￣￣|");
			Console.WriteLine("|  {0}   |  {1}   |  {2}   |", nums[1, 0], nums[1, 1], nums[1, 2]);
			Console.WriteLine("|      |      |      |");
			Console.WriteLine("|￣￣￣|￣￣￣|￣￣￣|");
			Console.WriteLine("|  {0}   |  {1}   |  {2}   |", nums[2, 0], nums[2, 1], nums[2, 2]);
			Console.WriteLine("|      |      |      |");
			Console.WriteLine(" ￣￣￣ ￣￣￣ ￣￣￣ ");
		}

	}

}
