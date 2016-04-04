using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MsTest
{
	class Astar
	{
		int width, height;
		int[] puzzle;
		public Astar(int _width, int _height, int[] _puzzle)
		{
			width = _width;
			height = _height;
			puzzle = _puzzle;
		}

		public void Solve()
		{
			//探索時間を計測するストップウォッチ
			System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
			sw.Start();

			//解答のステップ数
			int step = 0;
			//これから訪問する状態を入れておくオープンリスト
			Queue<State> openList = new Queue<State>();
			ISet<int> openLists = new HashSet<int>();
			//既に訪問済みの状態を入れておくクローズドリスト
			ISet<int> closedList = new HashSet<int>();

			//変数の初期設定
			int count = 0;
			int[,] firstNums = new int[height, width];
			int firstZeroX = -1, firstZeroY = -1;

			//パズルを作る
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					firstNums[y, x] = puzzle[count++];
					if (firstNums[y, x] == 0)
					{
						firstZeroY = y;
						firstZeroX = x;
					}
				}
			}

			/*****************************************************
			*													 *
            *					A*アルゴリズム                   *         
            *													 *
			* ****************************************************/

			//初期状態オブジェクトを生成してclosedListとopenListに入れる
			State firstState = new State(null, -1, firstNums, firstZeroY, firstZeroX, -1);
			closedList.Add(firstState.hashCode());
			openList.Enqueue(firstState);
			openLists.Add(firstState.hashCode());

			//openListが空なら探索は失敗で実行を終了する
			while (openList.Count > 0)
			{
				step++;
				//openListの先頭データを取り出してnowStateに代入
				State nowState = openList.Dequeue();
				openLists.Remove(nowState.hashCode());

				//nowStateが目標状態状態なら移動順を列挙して終了
				//逆から辿っているのでひっくり返して出力
				if (nowState.hashCode() == 123456780)
				{
					//ストップウォッチを停止
					sw.Stop();
					LinkedList<String> moves = new LinkedList<String>();
					LinkedList<State> movess = new LinkedList<State>();
					Queue<String> moveq = new Queue<string>();
					while (nowState.preMove != -1)
					{
						moves.AddFirst(State.vName[nowState.preMove]);
						movess.AddFirst(nowState);
						nowState = nowState.preState;
					}
					foreach (String move in moves)
					{
						moveq.Enqueue(move);
					}
					//手順を表示する
					int loop = 0;
					firstState.showState();
					Console.WriteLine("↓move:" + moveq.Dequeue());
					Console.WriteLine();
					foreach (State move in movess)
					{
						move.showState();
						if (moveq.Count != 0)
							Console.WriteLine("↓move:" + moveq.Dequeue());
						Console.WriteLine();
						loop++;
					}

					Console.WriteLine("time:" + sw.ElapsedMilliseconds + "ms  step:" + step);
					System.Console.Out.WriteLine("done");
					Console.ReadLine();
					return;
				}

				//closedListにnowStateを追加
				closedList.Add(nowState.hashCode());

				//nowStateから移動できる状態をまだ調べてなければ
				//closedListとopenListに追加する
				foreach (State nextState in nowState.nextMove())
				{
					//小節点nextStateに対してコストを計算
					int gnm = nowState.g + 1;
					int fnm = gnm + nextState.h;

					//i)openListにもclosedListにもnextStateが含まれていない
					if (!closedList.Contains(nextState.hashCode()) && !openLists.Contains(nextState.hashCode()))
					{
						openList.Enqueue(nextState);
						openLists.Add(nextState.hashCode());
					}
					//ii)openListまたはclosedListに含まれている
					else if (closedList.Contains(nextState.hashCode()) || openLists.Contains(nextState.hashCode()))
					{
						//f*(n,ni)<f(ni)の場合
						if (fnm < nextState.cost)
						{
							//nextStateをclosedListから削除しopenListに入れる
							closedList.Remove(nextState.hashCode());
							openList.Enqueue(nextState);
							openLists.Add(nextState.hashCode());

							//nextStateの親節点をnowStateにする
							nextState.preState = nowState;
							nextState.setCost(gnm, nextState.h);
						}
					}
					//openList内の各節点をcostの値で昇順にソートする
					Queue<State> ordered = new Queue<State>(openList.OrderBy(o => o.cost));
					openList = ordered;
				}
			}

			//解が存在しなかった時の処理
			Console.WriteLine("no answer");
			Console.ReadLine();
			return;

		}
	}
}
